#pragma once

#include <cuda.h>
#include "device.hpp"
#include "error.hpp"

namespace cuda {

	namespace driver {

		class context_base {
		public:
			using context_type = CUcontext;

		protected:
			context_type m_ctx;
			const device &m_dev;

			context_base(CUcontext ctx, const device &dev) : m_ctx(ctx), m_dev(dev) {}
		};

		class context : public context_base {
		private:
			static context_type create_new(const device &dev) {
				context_type ctx;
				attempt(cuCtxCreate(&ctx, 0, dev.hdl()), "Failed to create context");
				return ctx;
			}

		public:
			context(const device &dev) : context_base(create_new(dev), dev) {
			}

			~context() {
				// TODO: how to deal with error?
				cuCtxDestroy(m_ctx);
			}
		};

		class context_ref : public context_base {
		private:
			static context_type get_current() {
				context_type ctx;
				attempt(cuCtxGetCurrent(&ctx), "Failed to create context");
				return ctx;
			}

		public:
			context_ref() : context_base(get_current(), device::current()) {
			}
		};

	} // namespace driver

} // namespace cuda