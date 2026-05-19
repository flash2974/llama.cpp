#pragma once
#include <string>

std::string preprocess(std::string data);

std::string loadStudent(const std::string& path);
void parse_json(std::string content);