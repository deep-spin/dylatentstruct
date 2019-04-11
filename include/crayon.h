#pragma once

/* MLFlow interaction */

#include <string>
#include <chrono>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

//#include<iostream>

using nlohmann::json;

struct Crayon
{
    std::string hostname;
    std::string port;
    std::string run_uuid;

    Crayon(
        const std::string& run_uuid,
        const std::string& hostname = std::string("localhost"),
        const std::string& port = std::string("8889"))
        : hostname(hostname)
        , port(port)
        , run_uuid(run_uuid)
    {
        //std::cerr << "run_uuid" << run_uuid << "\n";
        std::stringstream url;
        url << "http://" << hostname << ":" << port << "/data";
        json payload = run_uuid;
        auto r = cpr::Post(cpr::Url(url.str()),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});
        //std::cerr << r.text << "\n";

    }

    cpr::Url get_url(const std::string& name)
    {
        std::stringstream url;
        url << "http://" << hostname << ":" << port
            << "/data/scalars?xp=" << run_uuid << "&name=" << name;
        auto cpurl = cpr::Url(url.str());
        return cpurl;
    }

    void log_metric(const std::string& key, const double& val, const unsigned& step)
    {
        auto payload = json{seconds_since_epoch(), step, val};
        auto r = cpr::Post(get_url(key),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});

        //std::cerr << r.text << "\n";
    }

private:
    double seconds_since_epoch()
    {
        auto t = std::chrono::system_clock::now().time_since_epoch();
        auto tms = std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
        return tms / 1000;
    }
};
