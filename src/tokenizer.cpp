#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <climits>

namespace gpt2 {

void Tokenizer::init_byte_unicode_map() {
    std::vector<int> bs, cs;
    for (int i = '!'; i <= '~'; i++)    { bs.push_back(i); cs.push_back(i); }
    for (int i = 0xA1; i <= 0xAC; i++)  { bs.push_back(i); cs.push_back(i); }
    for (int i = 0xAE; i <= 0xFF; i++)  { bs.push_back(i); cs.push_back(i); }
    int n = 0;
    for (int i = 0; i < 256; i++) {
        bool found = false;
        for (int b : bs) if (b == i) { found = true; break; }
        if (!found) {
            bs.push_back(i);
            cs.push_back(256 + n++);
        }
    }
    for (size_t i = 0; i < bs.size(); i++) {
        byte_to_unicode_[static_cast<uint8_t>(bs[i])] = static_cast<char32_t>(cs[i]);
        unicode_to_byte_[static_cast<char32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
    }
}

static std::string cp_to_utf8(char32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

static std::string bytes_to_unicode_str(
    const std::string& text,
    const std::unordered_map<uint8_t, char32_t>& map) {
    std::string result;
    for (uint8_t b : text) result += cp_to_utf8(map.at(b));
    return result;
}

static bool parse_vocab_json(const std::string& json,
    std::unordered_map<std::string, int>& t2i,
    std::unordered_map<int, std::string>& i2t) {
    size_t pos = json.find('{');
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size()) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' ||
               json[pos] == '\r' || json[pos] == '\t' || json[pos] == ',')) pos++;
        if (pos >= json.size() || json[pos] == '}') break;
        if (json[pos] != '"') return false;
        pos++;
        std::string key;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case '"':  key += '"'; break;
                    case '\\': key += '\\'; break;
                    case 'n':  key += '\n'; break;
                    case 'r':  key += '\r'; break;
                    case 't':  key += '\t'; break;
                    case '/':  key += '/'; break;
                    case 'u': {
                        if (pos + 4 < json.size()) {
                            char32_t cp = std::stoul(json.substr(pos+1, 4), nullptr, 16);
                            key += cp_to_utf8(cp);
                            pos += 4;
                        }
                        break;
                    }
                    default: key += json[pos];
                }
            } else {
                key += json[pos];
            }
            pos++;
        }
        pos++;
        while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) pos++;
        int id = std::atoi(json.c_str() + pos);
        while (pos < json.size() && json[pos] != ',' && json[pos] != '}') pos++;
        t2i[key] = id;
        i2t[id] = key;
    }
    return !t2i.empty();
}

bool Tokenizer::load(const std::string& dir) {
    std::string d = dir;
    if (!d.empty() && d.back() != '/') d += '/';
    init_byte_unicode_map();

    std::ifstream vf(d + "vocab.json");
    if (!vf.is_open()) { std::cerr << "Cannot open vocab.json\n"; return false; }
    std::stringstream vs;
    vs << vf.rdbuf();
    if (!parse_vocab_json(vs.str(), token_to_id_, id_to_token_)) return false;
    std::cout << "Vocab: " << token_to_id_.size() << " tokens\n";

    std::ifstream mf(d + "merges.txt");
    if (!mf.is_open()) { std::cerr << "Cannot open merges.txt\n"; return false; }
    std::string line;
    std::getline(mf, line); // skip header
    int rank = 0;
    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        bpe_ranks_[{line.substr(0, sp), line.substr(sp + 1)}] = rank++;
    }
    std::cout << "Merges: " << bpe_ranks_.size() << " rules\n";
    return true;
}

std::vector<std::string> Tokenizer::bpe(const std::string& token) const {
    // Split into UTF-8 characters
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < token.size()) {
        uint8_t c = token[i];
        int len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        symbols.push_back(token.substr(i, len));
        i += len;
    }
    if (symbols.size() <= 1) return symbols;

    while (true) {
        // Find lowest-rank pair
        int best_rank = INT_MAX;
        std::pair<std::string, std::string> best;
        for (size_t j = 0; j + 1 < symbols.size(); j++) {
            auto it = bpe_ranks_.find({symbols[j], symbols[j+1]});
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best = it->first;
            }
        }
        if (best_rank == INT_MAX) break;

        // Merge all occurrences
        std::vector<std::string> merged;
        size_t j = 0;
        while (j < symbols.size()) {
            if (j + 1 < symbols.size() &&
                symbols[j] == best.first && symbols[j+1] == best.second) {
                merged.push_back(best.first + best.second);
                j += 2;
            } else {
                merged.push_back(symbols[j]);
                j++;
            }
        }
        symbols = std::move(merged);
        if (symbols.size() == 1) break;
    }
    return symbols;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    std::regex pat(R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+)");
    auto begin = std::sregex_iterator(text.begin(), text.end(), pat);
    auto end = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
        std::string word = (*it)[0].str();
        std::string unicode_word = bytes_to_unicode_str(word, byte_to_unicode_);
        for (const auto& tok : bpe(unicode_word)) {
            auto found = token_to_id_.find(tok);
            if (found != token_to_id_.end())
                ids.push_back(found->second);
        }
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) result += decode_token(id);
    return result;
}

std::string Tokenizer::decode_token(int id) const {
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) return "";

    std::string bytes;
    const std::string& token = it->second;
    size_t i = 0;
    while (i < token.size()) {
        uint8_t c = token[i];
        char32_t cp;
        int len = 1;
        if ((c & 0x80) == 0)      { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else                          { cp = c & 0x07; len = 4; }
        for (int j = 1; j < len && i + j < token.size(); j++)
            cp = (cp << 6) | (token[i+j] & 0x3F);
        i += len;
        auto found = unicode_to_byte_.find(cp);
        if (found != unicode_to_byte_.end())
            bytes += static_cast<char>(found->second);
    }
    return bytes;
}

} // namespace gpt2