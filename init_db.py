from database import engine
import models

# Create all tables
models.Base.metadata.create_all(bind=engine)

print("Database initialized successfully!")

