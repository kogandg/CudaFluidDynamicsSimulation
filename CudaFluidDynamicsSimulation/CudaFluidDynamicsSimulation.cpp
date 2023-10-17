#include <SFML/Graphics.hpp>
#include <iostream>
#include <chrono>
//#include <cstdlib>
//#include <cmath>

const int SCALE = 2;
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const int FIELD_WIDTH = (int)(WINDOW_WIDTH / SCALE);
const int FIELD_HEIGHT = (int)(WINDOW_HEIGHT / SCALE);


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
	bool bloomEnable;
} parameters;

void setParams(float vDiffusion = 0.8f, float pressure = 1.5f, float vorticity = 50.0f, float cDiffuion = 0.8f,
	float dDiffuion = 1.2f, float force = 1000.0f, float bloomIntesity = 25000.0f, int radius = 100, bool bloomEnable = true);
void computeField(uint8_t* result, float dt, int x1Pos, int y1Pos, int x2Pos, int y2Pos, bool isMousePressed);
void cudaInit(size_t xSize, size_t ySize);
void cudaExit();

int main()
{
	cudaInit(FIELD_WIDTH, FIELD_HEIGHT);
	//srand(time(NULL));

	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Fluid Dynamics", sf::Style::Close);

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

		window.draw(sprite);
		window.display();
	}
	cudaExit();
	return 0;
}