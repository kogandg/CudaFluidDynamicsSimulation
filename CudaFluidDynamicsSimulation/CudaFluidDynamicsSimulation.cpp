#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
#include <chrono>

#include "Parameters.h"

const int SCALE = 2;
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const int FIELD_WIDTH = (int)(WINDOW_WIDTH / SCALE);
const int FIELD_HEIGHT = (int)(WINDOW_HEIGHT / SCALE);

struct Setting
{
	const char* name;
	float* value;

	float step;

	Setting(const char* name, float* value, float step = 0.1)
	{
		this->name = name;
		this->value = value;
		this->step = step;
	}

	void increment(bool up)
	{
		if (up)
		{
			*(value) += step;
		}
		else if (*(value) > step)
		{
			*(value) -= step;
		}
	}
};

void computeField(uint8_t* result, float dt, int x1Pos, int y1Pos, int x2Pos, int y2Pos, bool isMousePressed);
void cudaInit(size_t xSize, size_t ySize, Parameters* params);
void cudaExit();

int main()
{
	Parameters* parameters = new Parameters();

	cudaInit(FIELD_WIDTH, FIELD_HEIGHT, parameters);
	//srand(time(NULL));

	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Fluid Dynamics", sf::Style::Close);

	sf::Font font;
	std::string fileName = "Tuffy.ttf";
	if (!font.loadFromFile(fileName))
	{
		return EXIT_FAILURE;
	}

	sf::Text settingsText;
	settingsText.setFont(font);
	settingsText.setCharacterSize(14);
	settingsText.setFillColor(sf::Color::White);
	settingsText.setOutlineColor(sf::Color::Black);
	settingsText.setOutlineThickness(2.0f);
	settingsText.setPosition(5.0f, 5.0f);

	std::ostringstream osstr;


	/*float velocityDiffusion;
	float pressure;
	float vorticity;
	float colorDiffusion;
	float densityDiffusion;
	float forceScale;
	float bloomIntensity;
	int radius;
	bool bloomEnable;*/

	Setting settings[] =
	{
		{"velocityDiffusion", &parameters->velocityDiffusion},
		{"pressure", &parameters->pressure},
		{"vorticity",          &parameters->vorticity, 1},
		{"colorDiffusion",          &parameters->colorDiffusion},
		{"densityDiffusion", &parameters->densityDiffusion},
		{"forceScale",       &parameters->forceScale, 10},
		{"bloomIntensity",        &parameters->bloomIntensity},
		{"radius",       &parameters->radius, 1},
	};
	const int settingCount = 8;
	int currentSetting = 0;

	auto start = std::chrono::system_clock::now();
	auto end = std::chrono::system_clock::now();

	sf::Texture texture;
	sf::Sprite sprite;
	std::vector<sf::Uint8> pixelBuffer(FIELD_WIDTH * FIELD_HEIGHT * 4);
	texture.create(FIELD_WIDTH, FIELD_HEIGHT);

	sf::Vector2i mousePosition1 = { -1, -1 };
	sf::Vector2i mousePosition2 = { -1, -1 };

	bool isMousePressed = false;
	bool isPaused = false;
	bool isEditableVisable = true;

	std::chrono::duration<float> timeDifference;
	sf::Event event;
	while (window.isOpen())
	{
		end = std::chrono::system_clock::now();
		timeDifference = end - start;
		start = end;

		window.clear(sf::Color::White);
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
			}

			if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
			{
				window.close();
			}

			if (event.type == sf::Event::KeyPressed)
			{
				if (event.key.code == sf::Keyboard::L)
				{
					isEditableVisable = !isEditableVisable;
				}
				if (isEditableVisable)
				{
					if (event.key.code == sf::Keyboard::Down)
					{
						currentSetting = (currentSetting + 1) % settingCount;
					}
					if (event.key.code == sf::Keyboard::Up)
					{
						currentSetting = (currentSetting + settingCount - 1) % settingCount;
					}
					if (event.key.code == sf::Keyboard::Left)
					{
						settings[currentSetting].increment(false);
					}
					if (event.key.code == sf::Keyboard::Right)
					{
						settings[currentSetting].increment(true);
					}
				}
			}

			if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Button::Left)
				{
					mousePosition1 = { event.mouseButton.x, event.mouseButton.y };
					mousePosition1.x /= SCALE;
					mousePosition1.y /= SCALE;
					isMousePressed = true;
				}
				else
				{
					isPaused = !isPaused;
				}
			}

			if (event.type == sf::Event::MouseButtonReleased)
			{
				isMousePressed = false;
			}

			if (event.type == sf::Event::MouseMoved)
			{
				std::swap(mousePosition1, mousePosition2);
				mousePosition2 = { event.mouseMove.x, event.mouseMove.y };
				mousePosition2.x /= SCALE;
				mousePosition2.y /= SCALE;
			}
		}

		float dt = 0.02f;
		if (!isPaused)
		{
			computeField(pixelBuffer.data(), dt, mousePosition1.x, mousePosition1.y, mousePosition2.x, mousePosition2.y, isMousePressed);
		}

		texture.update(pixelBuffer.data());
		sprite.setTexture(texture);
		sprite.setScale({ SCALE, SCALE });

		osstr.str("");
		if (isEditableVisable)
		{
			for (int i = 0; i < settingCount; i++)
			{
				auto name = settings[i].name;
				auto value = *(settings[i].value);
				osstr << ((i == currentSetting) ? ">>  " : "       ") << name << ":  " << value << "\n";
			}
			
		}
		settingsText.setString(osstr.str());

		window.draw(sprite);
		window.draw(settingsText);
		window.display();
	}
	cudaExit();
	return 0;
}