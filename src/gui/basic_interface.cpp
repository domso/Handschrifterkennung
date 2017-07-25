#include "basic_interface.h"

#include <SDL2/SDL.h>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>
#include "../data/sample.h"
#include <stdint.h>

namespace gui {

basic_interface::basic_interface(const int width, const int height, const int tile_width, const int tile_height) :
		m_width(width), m_height(height), m_sample_width(tile_width), m_sample_height(tile_height), m_tile_width(width / tile_width), m_tile_height(height / tile_height), m_output(tile_width, tile_height), m_active(true) {
}

bool basic_interface::init() {
	m_window = SDL_CreateWindow("Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, m_width, m_height, SDL_WINDOW_SHOWN);
	m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_SOFTWARE);
	return m_window != nullptr && m_renderer != nullptr;
}

void basic_interface::update() {
	data::sample<float> local_data(m_sample_width, m_sample_height);
	bool running = true;
	bool update = false;

	running &= clear();
	running &= draw_grid();

	while (running) {
		SDL_PumpEvents();
		int x, y;
		uint32_t mouseState = SDL_GetMouseState(&x, &y);
		if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT)) {
			int tile_x = x / m_tile_width;
			int tile_y = y / m_tile_height;
			if (tile_x < m_sample_width && tile_y < m_sample_height) {
				if (local_data[tile_y * m_sample_width + tile_x] != 1) {
					local_data[tile_y * m_sample_width + tile_x] = 1;

					draw_tile(tile_x, tile_y, 255);

					update = true;
				}
			}
		} else if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
			reset(local_data);
		}

		if (update) {
			update = !create_output(local_data);
		}

		ckeck_for_close(running);
		SDL_RenderPresent(m_renderer);
	}

	m_active = false;
}

bool basic_interface::is_active() const {
	return m_active;
}

void basic_interface::close() {
	SDL_DestroyWindow(m_window);
}

bool basic_interface::wait_for_output(data::sample<float>& output) {
	std::unique_lock < std::mutex > ul(m_mutex);

	while (!m_outputIsValid) {
		auto now = std::chrono::high_resolution_clock::now();
		if (std::cv_status::timeout == m_cond.wait_until(ul, now + std::chrono::seconds(1))) {
			return false;	//time out
		}
	}

	m_outputIsValid = false;
	output = m_output;

	return true;
}

bool basic_interface::draw_grid() {
	int result = 0;
	result += SDL_SetRenderDrawColor(m_renderer, 128, 128, 128, 255);
	for (int x = 0; x < m_width; x += m_tile_width) {
		result += SDL_RenderDrawLine(m_renderer, x, 0, x, m_height);
	}

	for (int y = 0; y < m_height; y += m_tile_height) {
		result += SDL_RenderDrawLine(m_renderer, 0, y, m_width, y);
	}

	return result == 0;
}

bool basic_interface::draw_tile(const int x, const int y, const uint8_t color) {
	int result = 0;
	SDL_Rect tile;
	tile.x = x * m_tile_width;
	tile.y = y * m_tile_height;
	tile.h = m_tile_height;
	tile.w = m_tile_width;
	result += SDL_SetRenderDrawColor(m_renderer, color, color, color, 255);
	result += SDL_RenderFillRect(m_renderer, &tile);

	return result == 0;
}

bool basic_interface::clear() {
	int result = 0;
	result += SDL_SetRenderDrawColor(m_renderer, 50, 50, 50, 255);
	result += SDL_RenderClear(m_renderer);

	return result == 0;
}

bool basic_interface::reset(data::sample<float>& s) {
	bool result = true;
	result &= clear();
	result &= draw_grid();

	for (auto& t : s.internal_data()) {
		t = 0;
	}

	return result;
}

void basic_interface::ckeck_for_close(bool& running) {
	SDL_Event events;
	while (SDL_PollEvent(&events)) {
		if (events.type == SDL_QUIT) {
			running = false;
			return;
		}
	}
}

bool basic_interface::create_output(data::sample<float>& local) {
	std::unique_lock < std::mutex > ul(m_mutex, std::try_to_lock);
	if (ul.owns_lock()) {
		m_output = local;
		m_outputIsValid = true;
		m_cond.notify_all();

		return true;
	}

	return false;
}
}
