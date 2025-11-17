from typing import Dict, Iterable, List, Optional, Tuple


class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
    - Los tokens iniciales son bytes individuales (0..255).
    - Durante el entrenamiento se obtienen los pares de tokens adyacentes más frecuentes y se fusionan, todo ello de forma iterativa.
    - La codificación (`encode`) aplica las fusiones aprendidas en orden.
    """

    def __init__(self):
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.vocab: Dict[Tuple[int, ...], int] = {}
        self.id2bytes: List[Tuple[int, ...]] = []

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        """
        Devuelve una lista de tokens como tuplas de bytes individuales
        """
        b = s.encode("utf-8")
        return [(x,) for x in b]

    @staticmethod
    def _count_pairs(lines_tokens: List[List[Tuple[int, ...]]]) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas
        """
        pair_counts: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        
        for line in lines_tokens:
            if len(line) < 2: 
                continue # No hay pares en líneas con menos de 2 tokens
                
            for i in range(len(line) - 1):
                pair = (line[i], line[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
        return pair_counts

    @staticmethod
    def _merge_in_line(line: List[Tuple[int, ...]],
                       pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)
        """
        if len(line) < 2:
            return line

        result: List[Tuple[int, ...]] = []
        i = 0
        
        while i < len(line):
            if i + 1 < len(line) and (line[i], line[i + 1]) == pair:
                merged = line[i] + line[i + 1]
                result.append(merged)
                i += 2
            else:
                result.append(line[i])
                i += 1
                
        return result

    def train(self, lines: Iterable[str], vocab_size: int = 1000, max_merges: Optional[int] = None):
        """
        Aprende las fusiones del BPE y construye los vocabularios.
        """
        lines_tokens = [self._to_byte_tokens(line) for line in lines]

        self.vocab = {}
        self.id2bytes = []
        
        for byte in range(256):
            token = (byte,)
            self.vocab[token] = len(self.vocab)
            self.id2bytes.append(token)

        byte_tokens = set()
        for line in lines_tokens:
            for token in line:
                if len(token) > 1:
                    byte_tokens.add(token)

            for token in sorted(byte_tokens):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                    self.id2bytes.append(token)

        if len(self.vocab) >= vocab_size:
            return
            
        n_merges = min(
            vocab_size - len(self.vocab),
            max_merges if max_merges is not None else float('inf')
        )
        
        for _ in range(n_merges):
 
            pair_counts = self._count_pairs(lines_tokens)
            if not pair_counts:
                break
                
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            self.merges.append(best_pair)
            
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)
            self.id2bytes.append(new_token)
            
            lines_tokens = [self._merge_in_line(line, best_pair) for line in lines_tokens]
            
            if len(self.vocab) >= vocab_size:
                break

    def encode(self, text: str) -> List[int]:
        """
        Convierte el texto de entrada en una lista de token IDs.
        """
        tokens = self._to_byte_tokens(text)
        
        for pair in self.merges:
            tokens = self._merge_in_line(tokens, pair)
        
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(self.vocab[token])
            else: # Descomponer en bytes individuales
                for byte in token:
                    result.append(self.vocab[(byte,)])
        return result

    def decode(self, ids: List[int]) -> str:
        """
        Convierte una lista de token IDs en texto.
        """
        byte_tokens = [self.id2bytes[id] for id in ids]
        
        all_bytes = []
        for token in byte_tokens:
            all_bytes.extend(token)
            
        return bytes(all_bytes).decode('utf-8') 
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto.
        """
        ids = self.encode(text)
        
        tokens = []
        for id in ids:
            hex_bytes = ' '.join(f'{b:02x}' for b in self.id2bytes[id])
            tokens.append(f'[{hex_bytes}]')
            
        return tokens 


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python p1_bpe.py train <input_train_corpus> <output_model_file>")
        print("  python p1_bpe.py eval <input_model_file> <input_text>")
        exit(1)
        
    command = sys.argv[1]
    
    if command == "train":
        if len(sys.argv) != 4:
            print("Uso para entrenamiento:")
            print("  python p1_bpe.py train <input_train_corpus> <output_model_file>")
            exit(1)
            
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        
        bpe = ByteLevelBPE()
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        bpe.train(lines)
        
        model_data = {
            'merges': [(list(p1), list(p2)) for p1, p2 in bpe.merges],
            'vocab': {','.join(str(b) for b in k): v for k, v in bpe.vocab.items()},
            'id2bytes': [list(b) for b in bpe.id2bytes]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
            
    elif command == "eval":
        if len(sys.argv) != 4:
            print("Uso para evaluación:")
            print("  python p1_bpe.py eval <input_model_file> <input_text>")
            exit(1)
            
        model_file = sys.argv[2]
        input_text = sys.argv[3]
        
        bpe = ByteLevelBPE()
        with open(model_file, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        bpe.merges = [(tuple(p1), tuple(p2)) for p1, p2 in model_data['merges']]
        bpe.vocab = {tuple(int(b) for b in k.split(',')): v for k, v in model_data['vocab'].items()}
        bpe.id2bytes = [tuple(b) for b in model_data['id2bytes']]
        
        tokens = bpe.tokenize(input_text)
        print("Tokens:", tokens)
        print("IDs:", bpe.encode(input_text))
        print("Texto reconstruido:", bpe.decode(bpe.encode(input_text)))
        
    else:
        print(f"Comando desconocido: {command}")
        print("Comandos válidos: train, eval")
        exit(1)
