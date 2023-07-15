import ast
import asyncio
import json
import traceback

import aiohttp as aiohttp
import requests
from django.http import HttpResponse, HttpResponseNotFound, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.template import Context
from django.template.loader import get_template
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from httplib2 import Http
from oauth2client import client, file, tools

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity

import os
import tensorflow_text
import tensorflow_hub as hub
import numpy as np

model_name = "IlyaGusev/fred_t5_ru_turbo_alpaca"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

creds = None
STORE = file.Storage('token.json')
SCOPES = "https://www.googleapis.com/auth/forms.body"
DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

API_TOKEN = None # YOUR HUGGING_FACE API
RUGPT3_API_URL = "https://api-inference.huggingface.co/models/ai-forever/rugpt3large_based_on_gpt2"
FALCON_API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
RU_EN_API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ru-en"
EN_RU_API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ru"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def index(request):
    return render(request, "index.html")


async def form(request):
    if request.method == 'POST':
        form_id = request.POST.get('form_id')
        ai_model = request.POST.get('ai_model')
        try:
            form_info = get_form(form_id)
            if ai_model == "fred_t5_ru_turbo_alpaca":
                answers = generate_answers(form_info)
            else:
                answers = await generate_answers_by_api(form_info, ai_model)
            return render(request, "form.html", context={'answers': answers, 'form_id': form_id, 'ai_model': ai_model,
                                                         'title': form_info['info']['documentTitle']})
        except HttpError:
            return HttpResponseNotFound(f"Гугл-формы с id {form_id} не существует")
        except Exception as e:
            if str(e):
                return HttpResponse(str(e), status=503)
            return HttpResponse("Error", status=500)
    return HttpResponse("Error", status=400)


def download(request):
    if request.method == "POST":
        form_id = request.POST.get('form_id')
        ai_model = request.POST.get('ai_model')
        title = request.POST.get('title')
        answers = ast.literal_eval(request.POST.get('answers'))
        data = {'title': title, 'ai_model': ai_model, 'form_id': form_id, 'questions': []}
        for i, (question, answer) in enumerate(answers):
            data['questions'].append(
                {'questionId': i + 1, 'question': question,
                 'answer': answer})
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        response = HttpResponse(json_data, content_type='application/json')
        response['Content-Disposition'] = 'attachment; filename="answers.json"'
        return response
    return HttpResponse("Error", status=400)


def get_form(formId):
    # generation_config.max_length = 128
    global creds
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets(os.path.join(os.getcwd(), 'credentials.json'), SCOPES)
        flags = tools.argparser.parse_args([])
        creds = tools.run_flow(flow, STORE, flags)
    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    return form_service.forms().get(formId=formId).execute()


async def query(payload, api_url):
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            return await response.json()


def generate(item):
    data = tokenizer('Вопрос: ' + item['title'] + '. Ответ: ', return_tensors="pt")
    data = {k: v for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    most_similar_text = get_most_similar_answer(answer, item)
    return item['title'], most_similar_text


async def generate_by_api(item, ai_model):
    if ai_model == "rugpt3large_based_on_gpt2":
        answer = await query({"inputs": item["title"], "options": {"wait_for_model": True}}, RUGPT3_API_URL)
        if "error" in answer:
            raise Exception(answer["error"])
        answer = answer[0]["generated_text"].replace(item["title"], "").strip()
    else:
        translated = await query({
            "inputs": item["title"], "options": {"wait_for_model": True}},
            RU_EN_API_URL)
        if "error" in translated:
            raise Exception(translated["error"])
        translated = translated[0]['translation_text']
        output = await query({
            "inputs": translated, "options": {"wait_for_model": True, }
        }, FALCON_API_URL)
        if "error" in output:
            raise Exception(output["error"])
        output = output[0]['generated_text'].replace(translated, "").strip()
        answer = await query({
            "inputs": output, "options": {"wait_for_model": True}},
            EN_RU_API_URL)
        if "error" in answer:
            raise Exception(answer["error"])
        answer = answer[0]['translation_text']
    most_similar_text = get_most_similar_answer(answer, item)
    return item['title'], most_similar_text


def get_most_similar_answer(answer, item):
    variants = [(item['title'] + ' ' + i['value'], i['value']) for i in
                item['questionItem']['question']['choiceQuestion']['options']]
    query_embedding = embed([answer])[0]
    text_embeddings = embed([i[0] for i in variants])
    similarity_scores = cosine_similarity([query_embedding], text_embeddings)[0]
    most_similar_index = np.argmax(similarity_scores)
    most_similar_text = variants[most_similar_index][1]
    return most_similar_text


def generate_answers(form_info):
    results = []
    for item in form_info['items']:
        if 'choiceQuestion' in item['questionItem']['question']:
            results.append(generate(item))
    return results


async def generate_answers_by_api(form_info, ai_model):
    tasks = []
    for item in form_info['items']:
        if 'choiceQuestion' in item['questionItem']['question']:
            tasks.append(asyncio.create_task(generate_by_api(item, ai_model)))
    return await asyncio.gather(*tasks)
