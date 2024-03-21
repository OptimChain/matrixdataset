from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import csv
import json
from django.views.decorators.csrf import csrf_exempt
import boto3
import pandas as pd
import sys
from django.core.paginator import EmptyPage, Paginator
import math


access_key  = '[redacted]'
secret_key = '[redacted'
bucket_name = "ledger-logging"
object_name = "[2023-11-15].csv"

def json_status_response(status_code, message):
    to_json = {
    "message": message
    }
    response = HttpResponse(json.dumps(to_json))
    response.status_code = status_code
    return response 


@csrf_exempt
def get_all_buckets(request):
    bucketnames = []
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3 = session.resource('s3')
    for bucket in s3.buckets.all():
        bucketnames.append(bucket.name)
    return JsonResponse(bucketnames, safe=False)


@csrf_exempt
def get_filenames(request):
    body_unicode = request.body.decode('utf-8')
    post_data = json.loads(body_unicode)
    session = boto3.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(post_data['bucket_name'])
    file_lists = []
    for my_bucket_object in my_bucket.objects.all():
        file_lists.append(my_bucket_object.key)
    return JsonResponse(file_lists, safe=False)


if sys.version_info[0] < 3: 
    from StringIO import StringIO # Python 2.x
else:
    from io import StringIO


def send_aws_request(bucket_name, object_name):
    s3_client = boto3.client("s3", aws_access_key_id=access_key, 
                             aws_secret_access_key=secret_key)

    
    # Read an object from the bucket
    response = s3_client.get_object(Bucket= bucket_name, 
                                    Key=object_name)
    return response


@csrf_exempt
def home(request):

    if request.body:
        body_unicode = request.body.decode('utf-8')
        post_data = json.loads(body_unicode)
        print(post_data)
        response = send_aws_request(bucket_name, post_data['object_name'])
        object_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(object_content))#[post_data['startInd']:post_data['endInd']]
        # new_df = df.to_json(orient='reco  rds', lines=True)
        new_df = convert_data_json(df)  
        PRODUCTS_PER_PAGE = 10000
        page = post_data['page_no']
        product_paginator = Paginator(new_df, PRODUCTS_PER_PAGE)
        try:
            datasets = product_paginator.page(page)
        except EmptyPage:
            datasets = product_paginator.page(product_paginator.num_pages)
        except:
            datasets = product_paginator.page(PRODUCTS_PER_PAGE)

        json_data = []
        json_data.append(
            {'page_obj':{'start':product_paginator.page_range.start,
                        'end':product_paginator.page_range.stop,
                        'current_idx':page}
                        })
        dataset_lst = []
        dataset_colms = []
        for data in datasets:
            if data['dataset'] not in dataset_colms:
                dataset_colms.append(data['dataset'])
            dataset_lst.append(data)
        json_data.append({"dataset":dataset_lst})
        json_data.append({'dataset_colms':dataset_colms})

        print(len(dataset_colms))
        if len(datasets) < 1:
            return json_status_response(400, "Schemas have not been matched.")
        return JsonResponse(json_data,
                            safe=False) 
    
    return HttpResponse('Home Page')


def convert_data_json(data):
    jsonArray = []
    for idx in range(len(data)):
        dateformat = pd.to_datetime(data['date'][idx])
        if type(data['value'][idx]) == str:
            return []
        else:
            jsonArray.append({
                'date':str(dateformat.date()),
                'dataset':data['dataset'][idx],
                'metric':data['metric'][idx],
                'value':data['value'][idx]
            })
    return jsonArray


@csrf_exempt
def get_dataset_data(request):
    response = send_aws_request(bucket_name, object_name)
    object_content = response["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(object_content))
    json_data = convert_data_json(df)
   # new_df = df.to_json(orient='records', lines=True)
    return JsonResponse(json_data, safe=False)
    # with open('media/_2023-11-15_.csv', mode ='r')as file:
    #     # reading the CSV file
    #     csvFile = csv.reader(file)
    #     jsonArray = convert_data_json(csvFile)
    # jsonString = json.dumps(jsonArray)
    # return JsonResponse(jsonArray[1:], safe=False)








