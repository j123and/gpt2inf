#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>
namespace gpt2 {

class Tokenizer {
public:
    bool load(const std::string& dir);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::string decode_token(int id) const;

private:
    std::vector<std::string> bpe(const std::string& token) const;

    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<int, std::string> id_to_token_;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;
    std::unordered_map<uint8_t, char32_t> byte_to_unicode_;
    std::unordered_map<char32_t, uint8_t> unicode_to_byte_;

    void init_byte_unicode_map();
};

} // namespace gpt2