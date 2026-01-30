import json

with open("processed_document.json", 'r', encoding="utf-8") as f:
    table_data = json.load(f)

tables = table_data["tables"]
file_path = table_data["document_metadata"]["file_path"]

new_docs = []
for table in tables:
    content_string = ""

    if table["metadata"]["table_title"] is not None:
        content_string += f"TABLE TITLE: {table["metadata"]["table_title"]}\n\n"
    
    headers = table["headers"]
    rows = table["rows"]

    for row in rows:
        for i in range(len(headers)):
            content_string += f"{headers[i]}: {row[i]}\n\n"

    
    table_title = "TABLE: " + table["table_id"]
    if table["metadata"]["table_title"] is not None:
        table_title = "TABLE: " + table["metadata"]["table_title"]
    
    obj = {"content": content_string, 
            "metadata": 
            {"heading_title": table_title, 
             "source": file_path}}

    new_docs.append(obj)

heading_file = "heading-chunks-w-title.json"

with open(heading_file, "r") as f:
    data = json.load(f)

data.extend(new_docs)

with open(heading_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

