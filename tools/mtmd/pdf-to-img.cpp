#include "pdf-to-img.h"

#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

inline bool string_starts_with(const std::string &s, const std::string &prefix) {
	return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

fs::path run_pdftoppm(const std::string& input_file) {
    fs::path filepath = fs::path(input_file);
    std::string prefix = filepath.stem().string();
    
    pid_t pid = fork();

    if (pid == 0) { // Enfant
        const char* args[] = {
            "pdftoppm", 
            "-png", 
            input_file.c_str(), 
            prefix.c_str(), 
            nullptr
        };
        
        execvp(args[0], const_cast<char**>(args));
        _exit(1);
    } 
    else if (pid > 0) { // Parent
        int status;
        waitpid(pid, &status, 0);
        if (!(WIFEXITED(status) && WEXITSTATUS(status) == 0)) {
            return "";
        }
    }
    
    return filepath;
}

void move_imgs(const fs::path& root, const std::string& prefix) {
    fs::path target_dir = root / prefix;

    if (!fs::exists(target_dir)) {
        fs::create_directories(target_dir);
    }

    for (const auto& entry : fs::directory_iterator(root)) {
        std::string filename = entry.path().filename().string();

        if (string_starts_with(filename, prefix) && 
            entry.path().extension() == ".png") {
            
            fs::path destination = target_dir / entry.path().filename();

            try {
                fs::rename(entry.path(), destination);
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Erreur move : " << e.what() << std::endl;
            }
        }
    }
}

void convert_and_move(const std::string& input_file) {
    fs::path path_info = run_pdftoppm(input_file);
    
    if (!path_info.empty()) {
        move_imgs(path_info.parent_path(), path_info.stem().string());
    }
}

int main(int argc, char* argv[]) {
    // 1. Vérification des arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <chemin_du_pdf>" << std::endl;
        return 1;
    }

    std::string input_pdf = argv[1];

    // 2. Vérification de l'existence du fichier
    if (!fs::exists(input_pdf)) {
        std::cerr << "Erreur : Le fichier '" << input_pdf << "' n'existe pas." << std::endl;
        return 1;
    }

    if (fs::path(input_pdf).extension() != ".pdf") {
        std::cerr << "Erreur : Le fichier doit être un .pdf" << std::endl;
        return 1;
    }

    std::cout << "Traitement de : " << input_pdf << "..." << std::endl;

    try {
        // 3. Lancement de la conversion et du déplacement
        // Cette fonction appelle run_pdftoppm puis move_imgs
        convert_and_move(input_pdf);

        fs::path p(input_pdf);
        std::cout << "\nSuccès !" << std::endl;
        std::cout << "Les images ont été placées dans : " 
                  << (p.parent_path() / p.stem()) << "/" << std::endl;
    } 
    catch (const std::exception& e) {
        std::cerr << "Une erreur est survenue : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}