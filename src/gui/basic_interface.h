#ifndef basic_interface_h
#define basic_interface_h

#include <SDL2/SDL.h>
#include <stdint.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "../data/sample.h"

namespace gui {
/**
 * SDL2-canvas-gui for handwriting
 * Supports hardware-acceleration
 */
class basic_interface {
public:

	/**
	 * Sets some init values
	 * @param width: x-resolution of the window
	 * @param height: y-resolution of the window
	 * @param tile_width: x-resolution of a sample
	 * @param tile_height: y-resolution of a sample
	 */
	basic_interface(const int width, const int height, const int tile_width, const int tile_height);
	/**
	 * Starts the SDL-Context and opens the window
	 * @return: true on success
	 */
	bool init();

	/**
	 * Main-Update-loop for the window
	 * Terminates after window was closed
	 */
	void update();

	/**
	 * @return: true, if the window is still open and active
	 */
	bool is_active() const;

	/**
	 * Closes the SDL-Context
	 */
	void close();

	/**
	 * Waits until a new sample was generated
	 * @return: true, if a new sample could be copied
	 */
	bool wait_for_output(data::sample<float>& output);
private:

	/**
	 * Draws a simple (tile_width x tile_height)-grid
	 * @return: true on success
	 */
	bool draw_grid();

	/**
	 * Draws a simple (tile_width x tile_height)-tile on (x,y) with the given color
	 * @param x: x-position of the left upper edge
	 * @param y: y-position of the left upper edge
	 * @param color: color for each RGB-channel
	 * @return: true on success
	 */
	bool draw_tile(const int x, const int y, const uint8_t color);

	/**
	 * Clears the screen
	 * @return: true on success
	 */
	bool clear();

	/**
	 * Resets the canvas
	 * @return: true on success
	 */
	bool reset(data::sample<float>& s);

	/**
	 *  Pops all window-events and set running to false, if the window was closed by the X-Button
	 *  @param running: output-variable
	 */
	void ckeck_for_close(bool& running);

	/**
	 *  Sets the given sample as the new generated sample
	 *  If any wait_for_output-call is currently accessing the
	 *  data, the function will return false immediately.
	 *  @param local: input sample
	 *  @return: true on success
	 */
	bool create_output(data::sample<float>& local);

	std::atomic<bool> m_active;
	std::mutex m_mutex;
	std::condition_variable m_cond;

	int m_width;
	int m_height;

	int m_tile_width;
	int m_tile_height;

	data::sample<float> m_output;bool m_outputIsValid = false;

	SDL_Window* m_window;
	SDL_Renderer* m_renderer;
};
}

#endif
