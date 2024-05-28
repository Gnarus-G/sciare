# Sciare

CLI tool to manage documents, do semantic searches through documents and prompting using the RAG (Retrieval Augmented Generation)
strategy to use relevant info from documents to extend the llm's inference. Using the llama2-uncensored model.

## Features

- [x] Llama Model
- [x] Upload
- [x] Download
- [x] Search (Semantic Search)
- [x] Ask (Inference)
- [ ] Chat Interface

## Install

```sh
git clone https://github.com/Gnarus-G/sciare.git
cd sciare
cargo install --path .
```

### Ollama

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

For more info (e.g building from source): https://github.com/ollama/ollama?tab=readme-ov-file#building

## Usage

```sh
ollama serve
```

```sh
sciare download https://static.lwn.net/images/pdf/LDD3/ch13.pdf
sciare search "USB Device Basics"
```

```
CLI tool to manage documents, doing semantic searches through documents, and prompting usign the RAG (Retrieval Augmented Generation) strategy to use relevant info from documents to extend the llm's strategy

Usage: sciare [OPTIONS] <COMMAND>

Commands:
  list        List all the documents in the database
  delete      Delete a document from the database
  upload      Upload a file to index
  download    Download a file from the internet
  search      Search across all the content saved
  ask         Ask a question, and get an answer that considers information across all the content saved
  config      Manage configuration values
  completion  Generate a completions file for a specified shell
  help        Print this message or the help of the given subcommand(s)

Options:
      --ollama-ip <OLLAMA_IP>  Ip address serving Ollama api, assuming port 11434
  -h, --help                   Print help
  -V, --version                Print version
```
