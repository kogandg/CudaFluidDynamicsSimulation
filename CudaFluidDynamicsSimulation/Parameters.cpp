#include "Parameters.h"

Parameters::Parameters()
{
	setParams();
}

void Parameters::setParams(float vDiffusion, float pressure, float vorticity, float cDiffuion, float dDiffuion, float force, float bloomIntensity, float radius, bool bloomEnabled)
{
	this->velocityDiffusion = vDiffusion;
	this->pressure = pressure;
	this->vorticity = vorticity;
	this->colorDiffusion = cDiffuion;
	this->densityDiffusion = dDiffuion;
	this->forceScale = force;
	this->bloomIntensity = bloomIntensity;
	this->radius = radius;
	this->bloomEnabled = bloomEnabled;
}
