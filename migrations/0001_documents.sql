CREATE TABLE IF NOT EXISTS document (
  name VARCHAR(250) PRIMARY KEY NOT NULL, 
  saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
  documentName VARCHAR(250) NOT NULL, 
  content TEXT NOT NULL, 
  saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  FOREIGN KEY (documentName) REFERENCES document(name)
);
