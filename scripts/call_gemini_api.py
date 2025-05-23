from google import genai


def call_gemini(key="..."):
    client = genai.Client(api_key=key)
    prompt = "Wypisz w słowniku, kto grał 3 główne postacie w filmie Matrix. Odpowiedź ogranicz do postaci jako klucza i imienia i nazwiska aktorów jako wartości. Pomiń komentarze."

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    print(response.text)


call_gemini()
