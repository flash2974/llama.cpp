#pragma once
#include <string>
#include <filesystem>
#include <vector>

std::filesystem::path run_pdftoppm(const std::string& input_file, int resolution = 150);
std::vector<std::filesystem::path> move_imgs(const std::filesystem::path& root, const std::string& prefix);
std::vector<std::filesystem::path> convert_and_move(const std::string& input_file);
