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

	std::atomic<bool> m_active;
	std::mutex m_mutex;
	std::condition_variable m_cond;

	int m_width;
	int m_height;

	int m_tile_width;
	int m_tile_height;

	data::sample<float> m_output;
	bool m_outputIsValid = false;

	SDL_Window* m_window;
	SDL_Renderer* m_renderer;
};

#endif
