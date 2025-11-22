
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
from urllib.parse import quote_plus

class Database:
    def __init__(self):
        uri = "mongodb+srv://skansal291:lLmDtxj4v4J7nkku@clusterml.qr74icy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterml"
        # Create a new client and connect to the server
        client = MongoClient(uri, server_api=ServerApi('1'))
        # connect to database
        db = client.get_database('carprediction')
        # connect to the collection
        self.records = db['salesprediction']
    def add_single_document(self, input_data):
        try:
            result = self.records.insert_one(input_data)
            return result
        except Exception as e:
            print("insert failed")
    def delete_all_documents(self):
        try:
            self.records.delete_many({})
        except Exception as e:
            print('Deletion failed')
    
if __name__ == '__main__':
    D = Database()
    D.delete_all_documents()




"""uri = "mongodb+srv://skansal291:lLmDtxj4v4J7nkku@clusterml.qr74icy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterml"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# connect to database
db = client.get_database('carprediction')
# connect to the collection
records = db['salesprediction']
# check the connection
print(records.count_documents({}))

df = pd.read_csv('CarSeats.csv')
# convert dataframe to records
examples = df.to_dict(orient='records')
print(examples)
records.delete_many({})
records.insert_many(examples)
print(records.count_documents({}))
print(list(records.find({'shelveloc':'Medium'})))
print(list(records.find({'Education':{'$lt':15}})))

result = list(records.find({
    '$and':[
        {
            'Urban':'yes'
        },
        {
            'Education':{'$gt':15}
        }
    ]
}))
print(result)
# updating documents
updated={'Urban':'good'}
result_2 = records.update_many({'Education':{'$lte':15}},{'$set':updated})
print(result_2.matched_count)
print(result_2.modified_count)
"""