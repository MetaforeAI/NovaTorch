#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Device.h>

class NovaHooks : public at::PrivateUse1HooksInterface {
public:
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return true; }
    bool hasPrimaryContext(c10::DeviceIndex /*device_index*/) const override {
        return true;
    }
    void init() const override {}
    at::Device getDeviceFromPtr(void* /*data*/) const override {
        return at::Device(c10::DeviceType::PrivateUse1, 0);
    }
};
