class Parameters
{
public:
	float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntensity;
	float radius;
	bool bloomEnabled;

	Parameters();

	void setParams(
		float vDiffusion = 0.8f,
		float pressure = 1.5f,
		float vorticity = 50.0f,
		float cDiffuion = 0.8f,
		float dDiffuion = 1.2f,
		float force = 5000.0f,
		float bloomIntensity = 0.1f,
		float radius = 400,
		bool bloomEnabled = true
	);
};

