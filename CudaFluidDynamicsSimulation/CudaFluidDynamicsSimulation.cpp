#include "SFML\Graphics.hpp"
#include <iostream>
using std::cout;

void addWithCuda(int* c, int* a, int* b, int size);

int main() {
    sf::RenderWindow window(sf::VideoMode(500, 500), "Hello SFML and CUDA!", sf::Style::Close);
    sf::Event event;

    unsigned long long size = 1 << 14; // 2^28
    int* a = new int[size];
    int* b = new int[size];
    int* c = new int[size];

    for (int i = 0; i < size; i++) {
        a[i] = i * 2;
        b[i] = i * 3;
    }

    while (window.isOpen()) {
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        addWithCuda(c, a, b, size);
        for (int i = 0; i < 20; i++) {
            cout << c[i] << " ";
        }
        cout << "\n";

        window.clear(sf::Color(0, 0, 0));
        window.display();
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}