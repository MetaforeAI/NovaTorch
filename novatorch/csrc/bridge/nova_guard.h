#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

/// DeviceGuard implementation for the Nova (PrivateUse1) backend.
///
/// Nova currently exposes a single Vulkan device (index 0) and a single
/// logical stream, so most methods are trivial pass-throughs.
class NovaDeviceGuardImpl final
    : public c10::impl::DeviceGuardImplInterface {
public:
    // -- type --
    c10::DeviceType type() const override {
        return c10::DeviceType::PrivateUse1;
    }

    // -- device management --
    c10::Device exchangeDevice(c10::Device d) const override {
        // Single device – old device is always index 0
        (void)d;
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    c10::Device getDevice() const override {
        return c10::Device(c10::DeviceType::PrivateUse1, 0);
    }

    void setDevice(c10::Device /*d*/) const override {
        // Single device – nothing to do
    }

    void uncheckedSetDevice(c10::Device /*d*/) const noexcept override {
        // Single device – nothing to do
    }

    // -- stream management --
    c10::Stream getStream(c10::Device d) const noexcept override {
        return c10::Stream(
            c10::Stream::DEFAULT,
            c10::Device(c10::DeviceType::PrivateUse1, d.index()));
    }

    c10::Stream getDefaultStream(c10::Device d) const override {
        return getStream(d);
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
        // Single stream – return the same default stream
        (void)s;
        return c10::Stream(
            c10::Stream::DEFAULT,
            c10::Device(c10::DeviceType::PrivateUse1, 0));
    }

    // -- device count --
    c10::DeviceIndex deviceCount() const noexcept override {
        return 1;
    }
};
