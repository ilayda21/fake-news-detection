import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/")

db = client["politifact"]
collection = db["politifact"]


def insert_documents(data):
    data_to_add = []
    for d in data:
        query = {"fact": d["fact"]}
        doc = collection.find(query)
        if doc.count() == 0:
            data_to_add.append(d)
    if len(data_to_add) > 0:
        inserted_docs = collection.insert_many(data_to_add)
        print(str(len(inserted_docs.inserted_ids)) + " data inserted")
    else:
        print(data_to_add)


def insert_document(data):
    inserted_doc = collection.insert(data)

