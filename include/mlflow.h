#pragma once

/* MLFlow interaction */

#include <string>
#include <chrono>

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

#define GIT_COMMIT "0"
#include "git_commit.h"

using nlohmann::json;

long milliseconds_since_epoch()
{
    auto t = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t).count();
}

struct MLFlowRun
{
    std::string hostname;
    std::string port;
    unsigned exp_id;

    std::string run_uuid;

    const std::string user_id = "vlad";

    MLFlowRun(
        unsigned exp_id,
        const std::string& hostname = std::string("localhost"),
        const std::string& port = std::string("5000"))
        : hostname(hostname)
        , port(port)
        , exp_id(exp_id)
    {

        auto payload = json{
            {"user_id", user_id},
            {"start_time", milliseconds_since_epoch()},
            {"experiment_id", exp_id},
            {"source_type", "PROJECT"},
            {"source_version", GIT_COMMIT}
        };

        auto r = cpr::Post(get_url("runs/create"),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});

        auto jr = json::parse(r.text);
        run_uuid = jr["run"]["info"]["run_uuid"];

        //set_tag("mlflow.source.type", "PROJECT");
        //set_tag("mlflow.source.git.commit", GIT_COMMIT);
    };

    cpr::Url get_url(const std::string& path)
    const
    {
        std::stringstream url;
        url << "http://" << hostname << ":" << port
            << "/api/2.0/preview/mlflow/" << path;
        auto cpurl = cpr::Url(url.str());
        return cpurl;
    }

    void set_tag(const std::string& key, const std::string& val)
    const
    {
        auto payload = json{
            {"run_uuid", run_uuid},
            {"key", key},
            {"value", val},
        };
        auto r = cpr::Post(get_url("runs/set-tag"),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});
    }

    void log_parameter(const std::string& key, const std::string& val)
    const
    {
        auto payload = json{
            {"run_uuid", run_uuid},
            {"key", key},
            {"value", val},
        };
        auto r = cpr::Post(get_url("runs/log-parameter"),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});
    }

    void log_metric(const std::string& key, const double& val)
    const
    {
        auto payload = json{
            {"run_uuid", run_uuid},
            {"timestamp", milliseconds_since_epoch()},
            {"key", key},
            {"value", val},
        };
        auto r = cpr::Post(get_url("runs/log-metric"),
                           cpr::Header{{"Content-Type", "application/json"}},
                           cpr::Body{payload.dump()});
    }

};
