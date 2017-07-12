#ifndef basic_interface_h
#define basic_interface_h

#include <SDL2/SDL.h>
#include <stdint.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "sample.h"

class basic_interface {
public:

	basic_interface(const int width, const int height, const int tile_width,
			const int tile_height);

	bool init();

	void update();

	bool is_active() const;

	void close();

	data::sample<float>& wait_for_output();

private:
	bool draw_grid();

	void clear();

	void wait_for_close(bool& running);

	bool create_output(data::sample<float>& local);

	void draw_tile(const int x, const int y, const uint8_t color);

	void reset(data::sample<float>& s);

	std::atomic<bool> active_;
	std::mutex mutex_;
	std::condition_variable cond_;

	int width_;
	int height_;

	int tile_width_;
	int tile_height_;

	data::sample<float> output_;
	bool output_valid_ = false;

	SDL_Window* window_;
	SDL_Renderer* renderer_;
};

#endif
