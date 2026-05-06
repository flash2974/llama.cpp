#include "pdf-to-img.h"

#include <unistd.h>
#include <sys/wait.h>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
namespace fs = filesystem;

inline bool string_starts_with(const string &s, const string &prefix) {
	return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

fs::path run_pdftoppm(const string& input_file, int resolution) {
    fs::path filepath = fs::absolute(input_file);
    fs::path folder = filepath.parent_path();
    string prefix = filepath.stem().string();
    
    string output_prefix = (folder / prefix).string();
    
    pid_t pid = fork();

    if (pid == 0) { // Enfant
        const char* args[] = {
            "pdftoppm",
            "-r",
            to_string(resolution).c_str(),
            "-png", 
            input_file.c_str(), 
            output_prefix.c_str(), 
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

vector<fs::path> move_imgs(const fs::path& root, const string& prefix) {
    fs::path target_dir = root / prefix;

    if (!fs::exists(target_dir)) {
        fs::create_directories(target_dir);
    }

    vector<fs::path> entries;
    for (const auto& entry : fs::directory_iterator(root)) {
        string filename = entry.path().filename().string();

        if (string_starts_with(filename, prefix) && 
            entry.path().extension() == ".png") {
            
            fs::path destination = target_dir / entry.path().filename();

            try {
                fs::rename(entry.path(), destination);
                entries.push_back(destination);
            } catch (const fs::filesystem_error& e) {
                cerr << "Erreur move : " << e.what() << endl;
            }
        }
    }
    
    auto get_num = [&prefix](const fs::path& p) {
        try {
            std::string stem = p.stem().string(); 
            std::string str_num = stem.substr(prefix.length() + 1); 
            return std::stoi(str_num);
        } catch (...) {
            return 0;
        }
    };

    // 3. Trier les entrées en mémoire avant le déplacement
    std::sort(entries.begin(), entries.end(), [&](const auto& a, const auto& b) {
        return get_num(a) < get_num(b);
    });
    return entries;
}

vector<fs::path> convert_and_move(const string& input_file) {
    fs::path path_info = run_pdftoppm(input_file);
    
    if (!path_info.empty()) {
        return move_imgs(path_info.parent_path(), path_info.stem().string());
    }
}

int main(int argc, char* argv[]) {
    // 1. Vérification des arguments
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <chemin_du_pdf>" << endl;
        return 1;
    }

    string input_pdf = argv[1];

    // 2. Vérification de l'existence du fichier
    if (!fs::exists(input_pdf)) {
        cerr << "Erreur : Le fichier '" << input_pdf << "' n'existe pas." << endl;
        return 1;
    }

    if (fs::path(input_pdf).extension() != ".pdf") {
        cerr << "Erreur : Le fichier doit être un .pdf" << endl;
        return 1;
    }

    cout << "Traitement de : " << input_pdf << "..." << endl;

    try {
        // 3. Lancement de la conversion et du déplacement
        // Cette fonction appelle run_pdftoppm puis move_imgs
        convert_and_move(input_pdf);

        fs::path p(input_pdf);
        cout << "\nSuccès !" << endl;
        cout << "Les images ont été placées dans : " 
                  << (p.parent_path() / p.stem()) << "/" << endl;
    } 
    catch (const exception& e) {
        cerr << "Une erreur est survenue : " << e.what() << endl;
        return 1;
    }

    return 0;
}