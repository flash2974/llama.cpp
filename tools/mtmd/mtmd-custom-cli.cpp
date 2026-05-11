#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "pdf-to-img.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>
#include <limits.h>
#include <cinttypes>
#include <clocale>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;
    llama_pos n_keep = 0; // fin du prompt statique

    common_debug_cb_user_data cb_data;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval;
        }
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break; // end of generation
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(), // output
                        &text, // text
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                ctx.lctx, // lctx
                chunks.ptr.get(), // chunks
                ctx.n_past, // n_past
                0, // seq_id
                ctx.n_batch, // n_batch
                true, // logits_last
                &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }

    ctx.n_past = new_n_past;

    LOG("\n");

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }
    ctx.n_keep = ctx.n_past;

    if (is_single_turn) {
        g_is_generating = true;
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = mtmd_default_marker() + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        for (const auto & image : params.image) {
            if (!ctx.load_media(image)) {
                return 1; // error is already printed by libmtmd
            }
        }
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        LOG("\n   /pdf <path>    load pdf");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");
        std::string content;
        bool is_first_response = true;
        std::vector<common_chat_msg> keep_chat_history;

        
        while (!g_is_interrupted) {
            /* autre que 1ere reponse : partie dynamique qui est supprimée a chaque fois
            KV : [STATIC][DYNAMIC]
                         ^--------^
                         partie supprimée à chaque tour   
            */
            if (!is_first_response) {
                // on delete le kv cache entre n_keep et le max (-1)
                llama_memory_seq_rm(llama_get_memory(ctx.lctx), 0, ctx.n_keep, -1);
                ctx.n_past = ctx.n_keep;
                ctx.chat_history = keep_chat_history; 
            }

            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            
            if (line.empty()) continue;
            if (line == "/quit" || line == "/exit") break;
            
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) return 1;
                ctx.n_keep = ctx.n_past;
                is_first_response = true;
                common_sampler_free(ctx.smpl);
                ctx.smpl = common_sampler_init(ctx.model, params.sampling);
                LOG("Chat history cleared\n\n");
                continue;
            }

            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            bool is_pdf   = line == "/pdf"   || line.find("/pdf ")   == 0;
            bool is_txt   = line == "/txt"   || line.find("/txt ")   == 0;
            
            if (is_image || is_audio || is_pdf || is_txt) {
                size_t prefix_len = (is_pdf || is_txt) ? 5 : 7;
                
                if (line.size() <= prefix_len) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                
                std::string media_path = string_strip(line.substr(prefix_len));

                if (is_txt) {
                    std::ifstream ifs(media_path);
                    if (!ifs.is_open()) {
                        LOG_ERR("ERR: Impossible de lire le fichier texte '%s'\n", media_path.c_str());
                    } else {
                        std::stringstream buffer;
                        buffer << ifs.rdbuf();
                        content = "Maintenant, voila la réponse d'un étudiant. Tu dois utiliser OBLIGATOIREMENT la clé \"table\". Pour chaque question trouvée, crée une clé au format \"Ex01\", \"Ex02\". Pour chaque question, l'objet doit contenir deux clés : \"grade\" et \"comment\". Pour la clé \"grade\" : Évalue l'exercice et donne une note. ATTENTION : Tu n'as le droit que d'utiliser soit des tiers (0/3, 1/3, 2/3, 3/3), soit des quarts (0/4, 1/4, 2/4, 3/4, 4/4). Choisis le ratio qui te semble le plus adapté à la taille de l'exercice. Pour la clé \"comment\" : Donne un commentaire bref mais clair sur l'exercice en question. N'oublie pas de faire TOUTES les réponses aux questions. Voici la réponse de l'étudiant :\n" + buffer.str();
                        
                        LOG("Text file '%s' loaded (%zu bytes)\n", media_path.c_str(), buffer.str().size());
                    }
                }
                else if (is_pdf) {
                    auto paths_vect = convert_and_move(media_path);
                    if (paths_vect.empty()) {
                        LOG_ERR("ERR: Impossible de convertir le PDF ou fichier introuvable : '%s'\n", media_path.c_str());
                    }
                    
                    for (const auto& entry : paths_vect) {
                        std::string fname_str = entry.string();
                        if (ctx.load_media(fname_str)) {
                            LOG("PDF page '%s' loaded as image\n", fname_str.c_str());
                            content += mtmd_default_marker(); 
                        } else {
                            LOG_ERR("ERR: Echec du chargement de l'image '%s'\n", fname_str.c_str());
                        }
                    }
                    content = "Tu es un professeur de C++ très exigeant. Tu t'apprêtes à corriger des copies. Voici le sujet de l'examen :\n" 
                              + content + 
                              "\nAnalyse ce document pour t'en imprégner. Réponds en une seule phrase rapide avec la clé \"understanding\".";
                }
                else if (ctx.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : "audio");
                    content += mtmd_default_marker();
                } else {
                    LOG_ERR("ERR: Echec du chargement du media '%s'\n", media_path.c_str());
                }
                // continue; // si on veut taper le prompt apres l'injection du texte
            }
            else {
                content += line;
            }

            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            
            int ret = eval_message(ctx, msg);
            if (ret) return 1;
            if (g_is_interrupted) break;

            /* a chaque nouveau prompt, on reset le sampler : le fichier de grammaire ayant 
            été "utilisé", on le réinjecte au modèle pour qu'il réponde à nouveau selon GBNF */
            common_sampler_free(ctx.smpl);
            ctx.smpl = common_sampler_init(ctx.model, params.sampling);

            if (generate_response(ctx, n_predict)) return 1;

            // le premier prompt est le prompt "statique"
            if (is_first_response) {
                ctx.n_keep = ctx.n_past;                // on save la taille du KV cache
                keep_chat_history = ctx.chat_history;   // et aussi l'historique
                is_first_response = false;
            }

            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
