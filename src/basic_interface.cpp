#include "basic_interface.h"

#include <SDL2/SDL.h>
#include <vector>
#include "sample.h"
#include <stdint.h>
#include <mutex>
#include <atomic>

basic_interface::basic_interface(const int width, const int height, const int tile_width,
			const int tile_height) :
			width_(width),
			height_(height),
			tile_width_(tile_width),
			tile_height_(tile_height),
			output_(0, tile_width, tile_height),
			active_(true)
{
}

bool basic_interface::init() {

	window_ = SDL_CreateWindow("Test", SDL_WINDOWPOS_UNDEFINED,
	SDL_WINDOWPOS_UNDEFINED, width_, height_, SDL_WINDOW_SHOWN);

	renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_ACCELERATED);
	return window_ != nullptr && renderer_ != nullptr;
}

void basic_interface::update() {
	data::sample<float> local_data(0, tile_width_, tile_height_);
	bool running = true;
	bool update = false;

	clear();
	draw_grid();

	while (running) {
		SDL_PumpEvents();
		int x, y;
		uint32_t mouseState = SDL_GetMouseState(&x, &y);
		if (mouseState & SDL_BUTTON(SDL_BUTTON_LEFT)) {
			int tile_x = x / tile_width_;
			int tile_y = y / tile_height_;
			if (local_data[tile_y * tile_width_ + tile_x] != 1) {
				local_data[tile_y * tile_width_ + tile_x] = 1;

				draw_tile(tile_x, tile_y, 255);

				update = true;
			}
		} else if (mouseState & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
			reset(local_data);
		}

		if (update) {
			update = !create_output(local_data);
		}
		wait_for_close(running);
		SDL_RenderPresent(renderer_);
	}

	active_ = false;
	cond_.notify_all();
}

bool basic_interface::is_active() const {
	return active_;
}

void basic_interface::close() {
	SDL_DestroyWindow(window_);
}

data::sample<float>& basic_interface::wait_for_output() {
	std::unique_lock <std::mutex> ul(mutex_);

	while (!output_valid_) {
		cond_.wait(ul);
	}

	output_valid_ = false;

	return output_;
}

bool basic_interface::draw_grid() {
	int result = 0;
	SDL_SetRenderDrawColor(renderer_, 128, 128, 128, 255);
	for (int x = 0; x < width_; x += tile_width_) {
		result += SDL_RenderDrawLine(renderer_, x, 0, x, height_);
	}

	for (int y = 0; y < height_; y += tile_height_) {
		result += SDL_RenderDrawLine(renderer_, 0, y, width_, y);
	}

	return result == 0;
}

void basic_interface::clear() {
	SDL_SetRenderDrawColor(renderer_, 50, 50, 50, 255);
	SDL_RenderClear(renderer_);
}

void basic_interface::wait_for_close(bool& running) {
	SDL_Event events;
	while (SDL_PollEvent(&events)) {
		if (events.type == SDL_QUIT) {
			running = false;
			return;
		}
	}
}

bool basic_interface::create_output(data::sample<float>& local) {
	std::unique_lock <std::mutex> ul(mutex_, std::try_to_lock);
	if (ul.owns_lock()) {
		output_ = local;
		output_valid_ = true;
		cond_.notify_all();
		return true;
	}

	return false;
}

void basic_interface::draw_tile(const int x, const int y, const uint8_t color) {
	SDL_Rect tile;
	tile.x = x * tile_width_;
	tile.y = y * tile_height_;
	tile.h = tile_height_;
	tile.w = tile_width_;
	SDL_SetRenderDrawColor(renderer_, color, color, color, 255);
	SDL_RenderFillRect(renderer_, &tile);
}

void basic_interface::reset(data::sample<float>& s) {
	clear();
	draw_grid();

	for (auto& t : s.internalData()) {
		t = 0;
	}
}

