#include "SFML\Graphics.hpp"
#include <iostream>

#include <chrono>
//#include <ctime>  

const int SCALE = 2;
const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const int FIELD_WIDTH = WINDOW_WIDTH / SCALE;
const int FIELD_HEIGHT = WINDOW_HEIGHT / SCALE;

//void setParams(float vDiffusion = 0.8f, float pressure = 1.5f, float vorticity = 50.0f, float cDiffusion = 0.8f,
//	float dDiffusion = 1.2f, float force = 1000.0f, float bloomIntensity = 25000.0f, int radius = 100, bool bloomEnable = true);

void cudaInit(size_t x, size_t y);
void cudaExit();

void computeField(unsigned char* result, float dt, int x1pos = -1, int y1pos = -1, int x2pos = -1, int y2pos = -1, bool isPressed = false);

int main() {
	cudaInit(FIELD_WIDTH, FIELD_HEIGHT);

	sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT), "Fluid Dynamics", sf::Style::Close);

	auto startTime = std::chrono::system_clock::now();
	auto endTime = std::chrono::system_clock::now();

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
		endTime = std::chrono::system_clock::now();
		timeDifference = endTime - startTime;
		startTime = endTime;

		window.clear(sf::Color::White);
		while (window.pollEvent(event)) 
		{
			if (event.type == sf::Event::Closed) 
			{
				window.close();
			}

			if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Button::Left)
				{
					mousePosition1 = { event.mouseButton.x, event.mouseButton.y };
					mousePosition1 /= SCALE;
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
				mousePosition2 /= SCALE;
			}
		}

		float dt = 0.02f;
		if (!isPaused)
		{
			computeField(pixelBuffer.data(), dt, mousePosition1.x, mousePosition2.y, mousePosition2.x, mousePosition2.y, isMousePressed);
		}

		texture.update(pixelBuffer.data());
		sprite.setTexture(texture);
		sprite.setScale({SCALE, SCALE});

		window.draw(sprite);
		window.display();
	}

	cudaExit();
	return 0;
}