import torch
import pickle
import os
 
from Model import GRU_API
from Config import Config

def generate_text(start_str="shall i compare thee to a summer's day?\n", gen_length=500, temperature=0.8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = os.path.join(Config.PROJECT_ROOT, "data")
    VOCAB_FILE = os.path.join(DATA_DIR, "vocab.pkl")
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "GRU_API_MODEL.pth")

    with open(VOCAB_FILE, 'rb') as f:
        char_to_ix, ix_to_char, vocab_size = pickle.load(f)

    model = GRU_API(
        vocab_size=vocab_size,
        embed_dim=Config.EMBED_DIM,
        num_hidden=Config.NUM_HIDDEN
    ).to(device)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型权重: {MODEL_SAVE_PATH}")

    model.eval()

    input_seq = [char_to_ix[ch] for ch in start_str]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1, device)

    generated_text = start_str
    with torch.no_grad():
        for _ in range(gen_length):
            output, hidden = model(input_tensor, hidden)
            output = output.squeeze(0).div(temperature).exp()
            top_i = torch.multinomial(output, 1)[0]
            predicted_char = ix_to_char[top_i.item()]
            generated_text += predicted_char
            input_tensor = torch.tensor([[top_i]], dtype=torch.long).to(device)
 
    print("\n--- 生成的文本 ---")
    print(generated_text)
    print("------------------\n")
 

if __name__ == '__main__':
    generate_text(start_str="from fairest creatures we desire increase,\n", gen_length=400, temperature=0.8)