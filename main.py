import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI

# .env-Datei laden
load_dotenv()

# Groq Client erstellen (OpenAI-kompatibel)
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# -------------------------------------------------
# 1. Strukturierte Antwort definieren (Pydantic)
# -------------------------------------------------
class Answer(BaseModel):
    """Strukturierte Antwort des Modells"""
    answer: str = Field(description="Die eigentliche Antwort auf die Frage")
    confidence: float = Field(
        description="Wie sicher ist das Modell? Wert zwischen 0.0 und 1.0",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Kurze Begründung der Antwort")


# -------------------------------------------------
# 2. Funktion, die eine Frage stellt und validierte Antwort zurückgibt
# -------------------------------------------------
def ask_question(question: str) -> Answer:
    """
    Stellt eine Frage an Groq und gibt eine validierte, strukturierte Antwort zurück.
    """
    prompt = f"""
    Beantworte die folgende Frage klar und ehrlich.
    Wenn du die Antwort nicht weißt, sage das offen.

    Frage: {question}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Du antwortest immer nur mit gültigem JSON gemäß dem vorgegebenen Schema."},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    **Answer.model_json_schema(),
                    "additionalProperties": False,
                },
            },
        },
        temperature=0.2,
    )

    # Antwort in das Pydantic-Modell umwandeln und validieren
    return Answer.model_validate_json(response.choices[0].message.content)


# -------------------------------------------------
# 3. Ausprobieren
# -------------------------------------------------
if __name__ == "__main__":
    frage = "Was ist der Unterschied zwischen venv und Docker?"
    
    try:
        ergebnis = ask_question(frage)
        
        print("\n=== Strukturierte Antwort ===")
        print(f"Antwort:     {ergebnis.answer}")
        print(f"Konfidenz:   {ergebnis.confidence:.2f}")
        print(f"Begründung:  {ergebnis.reasoning}")
        
    except Exception as e:
        print(f"Fehler aufgetreten: {e}")