#include <iostream>
#include <regex>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>

#include "custom-utils.h"

using namespace std;
using ordered_json = nlohmann::ordered_json;

string preprocess(string data) {
    // pour retirer les commentaires // et /* */
    regex re1(R"(/\*[\s\S]*?\*/|//.*)");
    regex re2(R"(\s+)");
    // regex re3(R"(#)");

    string no_comment = regex_replace(data, re1, "");
    string no_space = regex_replace(no_comment, re2, " ");
    // string good_spaces = regex_replace(no_space, re3, "\n#");
    return no_space;
}

string loadStudent(const string& path) {
    string result{};
    try {
        if (filesystem::exists(path) && filesystem::is_directory(path)) {
            for (const auto& entry : filesystem::recursive_directory_iterator(path)) {
                ifstream ifs(entry.path());
                stringstream buffer;
                buffer << ifs.rdbuf();
                if (filesystem::is_regular_file(entry)) {
                    string relative_name = entry.path().lexically_relative(path).c_str();
                    const bool is_theory = entry.path().extension() == ".txt";
                    const string section = is_theory ? "THEORIE" : "PRATIQUE";
                    result = result 
                    + "######BEGIN FILE [" 
                    + section 
                    + "] \"" 
                    + relative_name 
                    + "\"\n" 
                    + buffer.str()
                    + "\n######END FILE [" 
                    + section 
                    + "] \"" 
                    + relative_name + "\"\n";

                    cout << "Loaded " << relative_name << endl;
                }
            }
        } else {
            cerr << "Le chemin spécifié n'existe pas ou n'est pas un dossier." << endl;
        }
    } catch (const filesystem::filesystem_error& e) {
        cerr << "Erreur : " << e.what() << endl;
    }
    return result;
}

void parse_json(string content) {
    size_t start = content.find_last_of("<channel|>") + 1;
    size_t end = content.find_last_of('}');

    if (start != string::npos && end != string::npos && end >= start) {
        content = content.substr(start, end - start + 1);
    } else {
        cerr << "[Erreur JSON] Aucun objet JSON détecté dans la réponse.\n";
        return;
    }

    try {
        ordered_json document = ordered_json::parse(content);
        
        if (document.contains("understanding")) {
            cout << "\nPlan de correction :\n";
            cout << "--------------------------------------------------------\n";
            for (const auto& [q_id, q_data] : document["understanding"].items()) {
                string title = q_data.value("title", "Titre inconnu");
                string points = q_data.value("points", "?");
                string resume = q_data.value("resume", "Pas de résumé");

                cout << "[" << left << setw(6) << q_id << "] " 
                          << title << " (" << points << " pts)\n"
                          << "         -> " << resume << "\n";
            }
            cout << "--------------------------------------------------------\n\n";
        }

        if (document.contains("table")) {
            cout << left 
                      << setw(8)  << "Ex" 
                      << "| " << setw(9) << "Note" 
                      << "| " << setw(6) << "Pts" 
                      << "| " << setw(6) << "Total" 
                      << "| Commentaire\n";
            cout << "--------------------------------------------------------------------------------\n";

            double student_grade = 0.0;
            double total_grade = 0.0;

            for (const auto& [nom_exo, data] : document["table"].items()) {
                string note = data.value("grade", "N/A");
                
                double points = 0.0;
                try {
                    points = stod(data.value("points", "0"));
                } catch(...) {} 

                double weighted_points = 0.0;
                if (note != "N/A") {
                    float n = 0, t = 1;
                    if (sscanf(note.c_str(), "%f/%f", &n, &t) == 2 && t != 0) {
                        weighted_points = points * (n / t);
                        student_grade += weighted_points;
                        total_grade += points;
                    }
                }
                
                string commentaire = data.value("comment", "");
                cout << left 
                          << setw(8)  << nom_exo 
                          << "| " << setw(9) << note 
                          << "| " << setw(6) << fixed << setprecision(2) << points
                          << "| " << setw(6) << fixed << setprecision(2) << weighted_points
                          << "| " << commentaire << "\n";
            }
            
            double note_sur_20 = (total_grade > 0) ? (student_grade * 20.0) / total_grade : 0.0;
            
            cout << "\n================================================================================\n";
            cout << "NOTE FINALE : " << fixed << setprecision(2) << student_grade 
                      << " / " << total_grade << " (Soit " << note_sur_20 << " / 20)\n";
            cout << "================================================================================\n";
        }

    } catch (const nlohmann::json::exception& e) {
        cerr << "[Erreur JSON] Impossible de parser le texte : " << e.what() << "\n";
    }
}