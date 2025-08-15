# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 20:21:04 2025

@author: freez
"""

import openai
client = openai.OpenAI(api_key="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")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say hello"}]
)
print(response.choices[0].message.content)