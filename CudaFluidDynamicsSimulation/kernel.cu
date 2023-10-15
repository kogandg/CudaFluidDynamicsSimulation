#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Vector2
{
	float x = 0.0f;
	float y = 0.0f;

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

static struct SystemConfig
{
	int velocityIterations = 20;
	int pressureIterations = 40;
	int xThreads = 80;
	int yThreads = 1;
} systemConfig;


//put these in a function to live there
static Particle* newField;
static Particle* oldField;
static size_t xSize;
static size_t ySize;
static unsigned char colorField;
static float* oldPressure;
static float* newPressure;
static float* vorticityField;
static Color currentColor;
static float elapsedTime = 0.0f;
static float timeSincePress = 0.0f;


void cudaInit(size_t x, size_t y)
{
	//setParams


}


//computes curl of velocity field
__device__ float curl(Particle* field, size_t xSize, size_t ySize, int x, int y)
{
	Vector2 curl = field[y * xSize + x].velocity;

	float x1 = -curl.x;
	float x2 = -curl.x;
	float y1 = -curl.y;
	float y2 = -curl.y;

	if (x + 1 < xSize && x + 1 >= 0 && y < ySize && y >= 0) x1 = field[y * xSize + (x + 1)].velocity.x;
	if (x - 1 < xSize && x - 1 >= 0 && y < ySize && y >= 0) x1 = field[y * xSize + (x - 1)].velocity.x;
	if (x < xSize && x >= 0 && y + 1 < ySize && y + 1 >= 0) y1 = field[(y + 1) * xSize + x].velocity.y;
	if (x < xSize && x >= 0 && y - 1 < ySize && y - 1 >= 0) y1 = field[(y - 1) * xSize + x].velocity.y;

	float result = ((y1 - y2) - (x1 - x2)) * 0.05f;
	return result;

}

// computes vorticity field which should be passed to applyVorticity function
__global__ void computeVorticity(float* vField, Particle* field, size_t xSize, size_t ySize)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//vorticityField[y * xSize + x] = curl(field, xSize, ySize, x, y);
}

// main, calls vorticity -> diffusion -> force -> pressure -> project -> advect -> paint -> bloom
void computeField(unsigned char* result, float dt, int x1pos, int y1pos, int x2pos, int y2pos, bool isPressed)
{
	dim3 threadsPerBlock(systemConfig.xThreads, systemConfig.yThreads);
	dim3 numBlocks(xSize / threadsPerBlock.x, ySize / threadsPerBlock.y);

	// curls and vortisity
	computeVorticity << <numBlocks, threadsPerBlock >> > (vorticityField, oldField, xSize, ySize);
}