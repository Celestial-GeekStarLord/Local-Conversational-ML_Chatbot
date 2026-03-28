# chat_pretrained.py
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

SAVE_DIR = "Model_Pretrained/blenderbot"

print("Loading model...")
tokenizer = BlenderbotTokenizer.from_pretrained(SAVE_DIR)
model = BlenderbotForConditionalGeneration.from_pretrained(SAVE_DIR)
model.eval()

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Running on: {device}")


# Conversation history (BlenderBot supports multi-turn context)
conversation_history = []

def generate_response(user_input, max_history_turns=5):
    # Add user message to history
    conversation_history.append(user_input)

    # Build context: last N turns joined by tokenizer's separator
    context = "  ".join(conversation_history[-max_history_turns:])

    inputs = tokenizer(
        [context],
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        reply_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            min_length=5,
            num_beams=4,
            temperature=0.7,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()

    # Add bot response to history for next turn
    conversation_history.append(response)

    return response


print("=" * 40)
print("  BlenderBot Chatbot  |  type 'quit' to exit")
print("=" * 40)

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit", "bye"):
        print("Bot: Bye! Take care.")
        break

    response = generate_response(user_input)
    print(f"Bot: {response}")