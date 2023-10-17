#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "cudart_platform.h"

#include <algorithm>
#include <iostream>
#include <cstdlib>

using uint8_t = unsigned char;

struct Vector2
{
	float x = 0.0, y = 0.0;

	__device__ Vector2 operator-(Vector2 other)
	{
		Vector2 result;
		result.x = this->x - other.x;
		result.y = this->y - other.y;
		return result;
	}

	__device__ Vector2 operator+(Vector2 other)
	{
		Vector2 result;
		result.x = this->x + other.x;
		result.y = this->y + other.y;
		return result;
	}

	__device__ Vector2 operator*(float d)
	{
		Vector2 result;
		result.x = this->x * d;
		result.y = this->y * d;
		return result;
	}
};

struct Color
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	__host__ __device__ Color operator+ (Color other)
	{
		Color result;
		result.R = this->R + other.R;
		result.G = this->G + other.G;
		result.B = this->B + other.B;
		return result;
	}

	__host__ __device__ Color operator* (float d)
	{
		Color result;
		result.R = this->R * d;
		result.G = this->G * d;
		result.B = this->B * d;
		return result;
	}
};

struct Particle
{
	Vector2 velocity;
	Color color;
};

static struct Parameters
{
	float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntesity;
	int radius;
	bool bloomEnabled;
} parameters;

static struct SystemConfig
{
	int velocityIterations = 20;
	int pressureIterations = 40;
	int xThreads = 80;
	int yThreads = 1;
} sysConfig;

void setParams(
	float vDiffusion = 0.8f,
	float pressure = 1.5f,
	float vorticity = 50.0f,
	float cDiffuion = 0.8f,
	float dDiffuion = 1.2f,
	float force = 5000.0f,
	float bloomIntesity = 0.1f,
	int radius = 400,
	bool bloomEnabled = true
)
{
	parameters.velocityDiffusion = vDiffusion;
	parameters.pressure = pressure;
	parameters.vorticity = vorticity;
	parameters.colorDiffusion = cDiffuion;
	parameters.densityDiffusion = dDiffuion;
	parameters.forceScale = force;
	parameters.bloomIntesity = bloomIntesity;
	parameters.radius = radius;
	parameters.bloomEnabled = bloomEnabled;
}

static const int colorArraySize = 7;
Color colorArray[colorArraySize];

static Particle* newField;
static Particle* oldField;
static uint8_t* colorField;
static size_t xSize;
static size_t ySize;
static float* oldPressure;
static float* newPressure;
static float* vorticityField;
static Color currentColor;
static float elapsedTime = 0.0f;
static float timeSincePress = 0.0f;

void cudaExit()
{
	cudaFree(colorField);
	cudaFree(oldField);
	cudaFree(newField);
	cudaFree(oldPressure);
	cudaFree(newPressure);
	cudaFree(vorticityField);
}

void CUDACall(cudaError_t result)
{
	if (result)
	{
		std::cout << "CUDA error = " << static_cast<unsigned int>(result) << std::endl;
		if (result == cudaErrorNoDevice)
		{
			std::cout << "No CUDA devices" << std::endl;
		}
		cudaExit();
		cudaDeviceReset();
		exit(99);
	}
}

void cudaInit(size_t x, size_t y)
{
	int cudaDevices = 0;
	CUDACall(cudaGetDeviceCount(&cudaDevices));

	setParams();

	colorArray[0] = { 1.0f, 0.0f, 0.0f };
	colorArray[1] = { 0.0f, 1.0f, 0.0f };
	colorArray[2] = { 1.0f, 0.0f, 1.0f };
	colorArray[3] = { 1.0f, 1.0f, 0.0f };
	colorArray[4] = { 0.0f, 1.0f, 1.0f };
	colorArray[5] = { 1.0f, 0.0f, 1.0f };
	colorArray[6] = { 1.0f, 0.5f, 0.3f };

	int index = rand() % colorArraySize;
	currentColor = colorArray[index];

	xSize = x;
	ySize = y;

	CUDACall(cudaMalloc(&colorField, xSize * ySize * 4 * sizeof(uint8_t)));
	CUDACall(cudaMalloc(&oldField, xSize * ySize * sizeof(Particle)));
	CUDACall(cudaMalloc(&newField, xSize * ySize * sizeof(Particle)));
	CUDACall(cudaMalloc(&oldPressure, xSize * ySize * sizeof(float)));
	CUDACall(cudaMalloc(&newPressure, xSize * ySize * sizeof(float)));
	CUDACall(cudaMalloc(&vorticityField, xSize * ySize * sizeof(float)));
}

__device__ float clampHelper(float value, float min, float max)
{
	float maxRes = fmax(min, value);
	return fmin(max, maxRes);
}

__device__ bool insideBounds(int x, int y, size_t xSize, size_t ySize)
{
	return x < xSize && x >= 0 && y < ySize && y >= 0;
}

//interpolate the quantity of grid cells (bilinear interpolation)
__device__ Particle interpolate(Vector2 vector, Particle* field, size_t xSize, size_t ySize)
{
	float x1 = (int)vector.x;
	float y1 = (int)vector.y;
	float x2 = (int)vector.x + 1;
	float y2 = (int)vector.y + 1;

	Particle q1;
	Particle q2;
	Particle q3;
	Particle q4;

	q1 = field[(int)clampHelper(y1, 0.0f, ySize - 1.0f) * xSize + (int)clampHelper(x1, 0.0f, xSize - 1.0f)];
	q2 = field[(int)clampHelper(y2, 0.0f, ySize - 1.0f) * xSize + (int)clampHelper(x1, 0.0f, xSize - 1.0f)];
	q3 = field[(int)clampHelper(y1, 0.0f, ySize - 1.0f) * xSize + (int)clampHelper(x2, 0.0f, xSize - 1.0f)];
	q4 = field[(int)clampHelper(y2, 0.0f, ySize - 1.0f) * xSize + (int)clampHelper(x2, 0.0f, xSize - 1.0f)];

	float t1 = (x2 - vector.x) / (x2 - x1);
	float t2 = (vector.x - x1) / (x2 - x1);
	Vector2 f1 = q1.velocity * t1 + q3.velocity * t2;
	Vector2 f2 = q2.velocity * t1 + q4.velocity * t2;
	Color C1 = q2.color * t1 + q4.color * t2;
	Color C2 = q2.color * t1 + q4.color * t2;
	float t3 = (y2 - vector.y) / (y2 - y1);
	float t4 = (vector.y - y1) / (y2 - y1);

	Particle result;
	result.velocity = f1 * t3 + f2 * t4;
	result.color = C1 * t3 + C2 * t4;

	return result;
}

//adds quantity to particles using bilinear interpolation
__global__ void advect(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float dDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float decay = 1.0f / (1.0f + dDiffusion * dt);
	Vector2 position = { x * 1.0f, y * 1.0f };

	Particle& oldParticle = oldField[y * xSize + x];

	//find new particle tracing where it came from
	Particle newParticle = interpolate(position - oldParticle.velocity * dt, oldField, xSize, ySize);
	newParticle.velocity = newParticle.velocity * decay;
	newParticle.color = newParticle.color * decay;
	/*newParticle.color.R = fmin(1.0f, pow(newParticle.color.R, 1.005f) * decay);
	newParticle.color.G = fmin(1.0f, pow(newParticle.color.G, 1.005f) * decay);
	newParticle.color.B = fmin(1.0f, pow(newParticle.color.B, 1.005f) * decay);*/
	newField[y * xSize + x] = newParticle;
}

//iteration of jacobi method on color grid field
__device__ Color jacobiColor(Particle* colorField, size_t xSize, size_t ySize, Vector2 position, Color B, float alpha, float beta)
{
	Color xU;
	Color xD;
	Color xL;
	Color xR;
	Color result;

	int x = position.x;
	int y = position.y;

	if (insideBounds(x, y - 1, xSize, ySize)) xU = colorField[(int)(y - 1) * xSize + (int)(x)].color;
	if (insideBounds(x, y + 1, xSize, ySize)) xD = colorField[(int)(y + 1) * xSize + (int)(x)].color;
	if (insideBounds(x - 1, y, xSize, ySize)) xL = colorField[(int)(y)*xSize + (int)(x - 1)].color;
	if (insideBounds(x + 1, y, xSize, ySize)) xR = colorField[(int)(y)*xSize + (int)(x + 1)].color;

	result = (xU + xD + xL + xR + B * alpha) * (1.0f / beta);
	return result;
}

//calculates color field diffusion
__global__ void computeColor(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float cDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Vector2 position = { x * 1.0f, y * 1.0f };
	Color color = oldField[y * xSize + x].color;
	float alpha = cDiffusion * cDiffusion / dt;
	float beta = 4.0f + alpha;

	newField[y * xSize + x].color = jacobiColor(oldField, xSize, ySize, position, color, alpha, beta);
}

//iteration of jacobi method on velocity grid field
__device__ Vector2 jacobiVelocity(Particle* field, size_t xSize, size_t ySize, Vector2 v, Vector2 B, float alpha, float beta)
{
	Vector2 vU = B * -1.0f;
	Vector2 vD = B * -1.0f;
	Vector2 vR = B * -1.0f;
	Vector2 vL = B * -1.0f;

	if (insideBounds(v.x, v.y - 1, xSize, ySize)) vU = field[(int)(v.y - 1) * xSize + (int)(v.x)].velocity;
	if (insideBounds(v.x, v.y + 1, xSize, ySize)) vD = field[(int)(v.y + 1) * xSize + (int)(v.x)].velocity;
	if (insideBounds(v.x - 1, v.y, xSize, ySize)) vL = field[(int)(v.y) * xSize + (int)(v.x - 1)].velocity;
	if (insideBounds(v.x + 1, v.y, xSize, ySize)) vR = field[(int)(v.y) * xSize + (int)(v.x + 1)].velocity;

	Vector2 result = (vU + vD + vL + vR + B * alpha) * (1.0f / beta);
	return result;
}

//calculates nonzero divergency velocity field u
__global__ void diffuse(Particle* newField, Particle* oldField, size_t xSize, size_t ySize, float vDiffusion, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Vector2 position = { x * 1.0f, y * 1.0f };
	Vector2 velocity = oldField[y * xSize + x].velocity;

	float alpha = vDiffusion * vDiffusion / dt;
	float beta = 4.0f + alpha;
	newField[y * xSize + x].velocity = jacobiVelocity(oldField, xSize, ySize, position, velocity, alpha, beta);
}

//performs several iterations over velocity and color fields
void computeDiffusion(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	//diffuse velocity and color
	for (int i = 0; i < sysConfig.velocityIterations; i++)
	{
		diffuse << <numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, parameters.velocityDiffusion, dt);
		computeColor << <numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, parameters.colorDiffusion, dt);
		std::swap(newField, oldField);
	}
}

//applies force and add color dye to the particle field
__global__ void applyForce(Particle* field, size_t xSize, size_t ySize, Color color, Vector2 F, Vector2 position, int radius, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float e = expf(-((x - position.x) * (x - position.x) + (y - position.y) * (y - position.y)) / radius);
	Vector2 uF = F * dt * e;

	Particle& particle = field[y * xSize + x];
	particle.velocity = particle.velocity + uF;

	color = color * e + particle.color;
	particle.color.R = fmin(1.0f, color.R);
	particle.color.G = fmin(1.0f, color.G);
	particle.color.B = fmin(1.0f, color.B);
}

//computes curl of velocity field
__device__ float curl(Particle* field, size_t xSize, size_t ySize, int x, int y)
{
	Vector2 C = field[int(y) * xSize + int(x)].velocity;

	float x1 = -C.x;
	float x2 = -C.x;
	float y1 = -C.y;
	float y2 = -C.y;

	if (insideBounds(x + 1, y, xSize, ySize)) x1 = field[y * xSize + (x + 1)].velocity.x;
	if (insideBounds(x - 1, y, xSize, ySize)) x2 = field[y * xSize + (x - 1)].velocity.x;
	if (insideBounds(x, y + 1, xSize, ySize)) y1 = field[(y + 1) * xSize + x].velocity.y;
	if (insideBounds(x, y - 1, xSize, ySize)) y2 = field[(y - 1) * xSize + x].velocity.y;

	float result = ((y1 - y2) - (x1 - x2)) * 0.5f;
	return result;
}

//computes absolute value gradient of vorticity field
__device__ Vector2 absGradient(float* field, size_t xSize, size_t ySize, int x, int y)
{
	float C = field[int(y) * xSize + int(x)];

	float x1 = C;
	float x2 = C;
	float y1 = C;
	float y2 = C;

	if (insideBounds(x + 1, y, xSize, ySize)) x1 = field[y * xSize + (x + 1)];
	if (insideBounds(x - 1, y, xSize, ySize)) x2 = field[y * xSize + (x - 1)];
	if (insideBounds(x, y + 1, xSize, ySize)) y1 = field[(y + 1) * xSize + x];
	if (insideBounds(x, y - 1, xSize, ySize)) y2 = field[(y - 1) * xSize + x];

	Vector2 result = { (abs(x1) - abs(x2)) * 0.5f, (abs(y1) - abs(y2)) * 0.5f };
	return result;
}

//computes vorticity field which should be passed to applyVorticity function
__global__ void computeVorticity(float* vField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	vField[y * xSize + x] = curl(field, xSize, ySize, x, y);
}

//applies vorticity to velocity field
__global__ void applyVorticity(Particle* newField, Particle* oldField, float* vField, size_t xSize, size_t ySize, float vorticity, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Particle& oldParticle = oldField[y * xSize + x];
	Particle& newParticle = newField[y * xSize + x];

	Vector2 v = absGradient(vField, xSize, ySize, x, y);
	v.y *= -1.0f;

	float length = sqrtf(v.x * v.x + v.y * v.y) + 1e-5f;
	Vector2 vNorm = v * (1.0f / length);

	Vector2 vF = vNorm * vField[y * xSize + x] * vorticity;
	newParticle = oldParticle;
	newParticle.velocity = newParticle.velocity + vF * dt;
}

//performs iteration of jacobi method on pressure grid field
__device__ float jacobiPressure(float* pressureField, size_t xSize, size_t ySize, int x, int y, float B, float alpha, float beta)
{
	float C = pressureField[int(y) * xSize + int(x)];

	float xU = C;
	float xD = C;
	float xL = C;
	float xR = C;

	if (insideBounds(x, y - 1, xSize, ySize)) xU = pressureField[(y - 1) * xSize + x];
	if (insideBounds(x, y + 1, xSize, ySize)) xD = pressureField[(y + 1) * xSize + x];
	if (insideBounds(x - 1, y, xSize, ySize)) xL = pressureField[y * xSize + (x - 1)];
	if (insideBounds(x + 1, y, xSize, ySize)) xR = pressureField[y * xSize + (x + 1)];

	float pressure = (xU + xD + xL + xR + alpha * B) * (1.0f / beta);
	return pressure;
}

//computes divergence of velocity field
__device__ float divergence(Particle* field, size_t xSize, size_t ySize, int x, int y)
{
	Particle& C = field[int(y) * xSize + int(x)];

	float x1 = -1 * C.velocity.x;
	float x2 = -1 * C.velocity.x;
	float y1 = -1 * C.velocity.y;
	float y2 = -1 * C.velocity.y;

	if (insideBounds(x + 1, y, xSize, ySize)) x1 = field[(int)(y)*xSize + (int)(x + 1)].velocity.x;
	if (insideBounds(x - 1, y, xSize, ySize)) x2 = field[(int)(y)*xSize + (int)(x - 1)].velocity.x;
	if (insideBounds(x, y + 1, xSize, ySize)) y1 = field[(int)(y + 1) * xSize + (int)(x)].velocity.y;
	if (insideBounds(x, y - 1, xSize, ySize)) y2 = field[(int)(y - 1) * xSize + (int)(x)].velocity.y;

	return (x1 - x2 + y1 - y2) * 0.5f;
}

// performs iteration of jacobi method on pressure field
__global__ void computePressureImpl(Particle* field, size_t xSize, size_t ySize, float* newPressure, float* oldPressure, float pressure, float dt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float div = divergence(field, xSize, ySize, x, y);

	float alpha = -1.0f * pressure * pressure;
	float beta = 4.0;
	newPressure[y * xSize + x] = jacobiPressure(oldPressure, xSize, ySize, x, y, div, alpha, beta);
}

//performs several iterations over pressure field
void computePressure(dim3 numBlocks, dim3 threadsPerBlock, float dt)
{
	for (int i = 0; i < sysConfig.pressureIterations; i++)
	{
		computePressureImpl << <numBlocks, threadsPerBlock >> > (oldField, xSize, ySize, newPressure, oldPressure, parameters.pressure, dt);
		std::swap(oldPressure, newPressure);
	}
}

//gradient of pressure field
__device__ Vector2 gradient(float* field, size_t xSize, size_t ySize, int x, int y)
{
	float C = field[y * xSize + x];

	float x1 = C;
	float x2 = C;
	float y1 = C;
	float y2 = C;

	if (insideBounds(x + 1, y, xSize, ySize)) x1 = field[y * xSize + (x + 1)];
	if (insideBounds(x - 1, y, xSize, ySize)) x2 = field[y * xSize + (x - 1)];
	if (insideBounds(x, y + 1, xSize, ySize)) y1 = field[(y + 1) * xSize + x];
	if (insideBounds(x, y - 1, xSize, ySize)) y2 = field[(y - 1) * xSize + x];

	Vector2 result = { (x1 - x2) * 0.5f, (y1 - y2) * 0.5f };
	return result;
}

// projects pressure field on velocity field
__global__ void project(Particle* newField, size_t xSize, size_t ySize, float* pressureField)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	Vector2& velocity = newField[y * xSize + x].velocity;
	velocity = velocity - gradient(pressureField, xSize, ySize, x, y);
}

//adds flashlight effect near the mouse position
__global__ void applyBloom(uint8_t* colorField, size_t xSize, size_t ySize, int xpos, int ypos, float radius, float bloomIntensity)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int position = 4 * (y * xSize + x);

	float e = bloomIntensity * expf(-((x - xpos) * (x - xpos) + (y - ypos) * (y - ypos) + 1.0f) / (radius * radius));

	uint8_t R = colorField[position + 0];
	uint8_t G = colorField[position + 1];
	uint8_t B = colorField[position + 2];

	uint8_t maxVal = fmaxf(R, fmaxf(G, B));

	colorField[position + 0] = fmin(255.0f, R + maxVal * e);
	colorField[position + 1] = fmin(255.0f, G + maxVal * e);
	colorField[position + 2] = fmin(255.0f, B + maxVal * e);
}


//fills output image with corresponding color
__global__ void paint(uint8_t* colorField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float R = field[y * xSize + x].color.R;
	float G = field[y * xSize + x].color.G;
	float B = field[y * xSize + x].color.B;

	colorField[4 * (y * xSize + x) + 0] = fmin(255.0f, 255.0f * R);
	colorField[4 * (y * xSize + x) + 1] = fmin(255.0f, 255.0f * G);
	colorField[4 * (y * xSize + x) + 2] = fmin(255.0f, 255.0f * B);
	colorField[4 * (y * xSize + x) + 3] = 255;
}

//main function, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void computeField(uint8_t* result, float dt, int x1Pos, int y1Pos, int x2Pos, int y2Pos, bool isPressed)
{
	dim3 threadsPerBlock(sysConfig.xThreads, sysConfig.yThreads);
	dim3 numBlocks(xSize / threadsPerBlock.x, ySize / threadsPerBlock.y);

	//curls and vortisity
	computeVorticity << <numBlocks, threadsPerBlock >> > (vorticityField, oldField, xSize, ySize);
	applyVorticity << <numBlocks, threadsPerBlock >> > (newField, oldField, vorticityField, xSize, ySize, parameters.vorticity, dt);
	std::swap(oldField, newField);

	//diffuse velocity and color
	computeDiffusion(numBlocks, threadsPerBlock, dt);

	//apply force
	if (isPressed)
	{
		timeSincePress = 0.0f;
		elapsedTime += dt;

		int roundTime = (int)(elapsedTime) % colorArraySize;
		int ceilTime = (int)((elapsedTime) + 1) % colorArraySize;

		float w = elapsedTime - (int)(elapsedTime);
		currentColor = colorArray[roundTime] * (1 - w) + colorArray[ceilTime] * w;

		Vector2 force;
		force.x = (x2Pos - x1Pos) * parameters.forceScale;
		force.y = (y2Pos - y1Pos) * parameters.forceScale;

		Vector2 position = { x2Pos * 1.0f, y2Pos * 1.0f };

		applyForce << <numBlocks, threadsPerBlock >> > (oldField, xSize, ySize, currentColor, force, position, parameters.radius, dt);
	}
	else
	{
		timeSincePress += dt;
	}

	//compute pressure
	computePressure(numBlocks, threadsPerBlock, dt);

	//project
	project << <numBlocks, threadsPerBlock >> > (oldField, xSize, ySize, oldPressure);
	CUDACall(cudaMemset(oldPressure, 0, xSize * ySize * sizeof(float)));

	//advect
	advect << <numBlocks, threadsPerBlock >> > (newField, oldField, xSize, ySize, parameters.densityDiffusion, dt);
	std::swap(newField, oldField);

	//paint image
	paint << <numBlocks, threadsPerBlock >> > (colorField, oldField, xSize, ySize);

	//apply bloom in mouse pos
	if (parameters.bloomEnabled && timeSincePress < 5.0f)
	{
		applyBloom << <numBlocks, threadsPerBlock >> > (colorField, xSize, ySize, x2Pos, y2Pos, parameters.radius, parameters.bloomIntesity);
	}

	//copy image to cpu
	size_t size = xSize * ySize * 4 * sizeof(uint8_t);
	CUDACall(cudaMemcpy(result, colorField, size, cudaMemcpyDeviceToHost));
}