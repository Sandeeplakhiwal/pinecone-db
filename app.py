from pinecone import Pinecone
from dotenv import load_dotenv
import os
import time


data = [
    { "id": "rec1", "chunk_text": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history" },
    { "id": "rec2", "chunk_text": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science" },
    { "id": "rec3", "chunk_text": "Albert Einstein developed the theory of relativity.", "category": "science" },
    { "id": "rec4", "chunk_text": "The mitochondrion is often called the powerhouse of the cell.", "category": "biology" },
    { "id": "rec5", "chunk_text": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature" },
    { "id": "rec6", "chunk_text": "Water boils at 100°C under standard atmospheric pressure.", "category": "physics" },
    { "id": "rec7", "chunk_text": "The Great Wall of China was built to protect against invasions.", "category": "history" },
    { "id": "rec8", "chunk_text": "Honey never spoils due to its low moisture content and acidity.", "category": "food science" },
    { "id": "rec9", "chunk_text": "The speed of light in a vacuum is approximately 299,792 km/s.", "category": "physics" },
    { "id": "rec10", "chunk_text": "Newton’s laws describe the motion of objects.", "category": "physics" },
    { "id": "rec11", "chunk_text": "The human brain has approximately 86 billion neurons.", "category": "biology" },
    { "id": "rec12", "chunk_text": "The Amazon Rainforest is one of the most biodiverse places on Earth.", "category": "geography" },
    { "id": "rec13", "chunk_text": "Black holes have gravitational fields so strong that not even light can escape.", "category": "astronomy" },
    { "id": "rec14", "chunk_text": "The periodic table organizes elements based on their atomic number.", "category": "chemistry" },
    { "id": "rec15", "chunk_text": "Leonardo da Vinci painted the Mona Lisa.", "category": "art" },
    { "id": "rec16", "chunk_text": "The internet revolutionized communication and information sharing.", "category": "technology" },
    { "id": "rec17", "chunk_text": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history" },
    { "id": "rec18", "chunk_text": "Dogs have an incredible sense of smell, much stronger than humans.", "category": "biology" },
    { "id": "rec19", "chunk_text": "The Pacific Ocean is the largest and deepest ocean on Earth.", "category": "geography" },
    { "id": "rec20", "chunk_text": "Chess is a strategic game that originated in India.", "category": "games" },
    { "id": "rec21", "chunk_text": "The Statue of Liberty was a gift from France to the United States.", "category": "history" },
    { "id": "rec22", "chunk_text": "Coffee contains caffeine, a natural stimulant.", "category": "food science" },
    { "id": "rec23", "chunk_text": "Thomas Edison invented the practical electric light bulb.", "category": "inventions" },
    { "id": "rec24", "chunk_text": "The moon influences ocean tides due to gravitational pull.", "category": "astronomy" },
    { "id": "rec25", "chunk_text": "DNA carries genetic information for all living organisms.", "category": "biology" },
    { "id": "rec26", "chunk_text": "Rome was once the center of a vast empire.", "category": "history" },
    { "id": "rec27", "chunk_text": "The Wright brothers pioneered human flight in 1903.", "category": "inventions" },
    { "id": "rec28", "chunk_text": "Bananas are a good source of potassium.", "category": "nutrition" },
    { "id": "rec29", "chunk_text": "The stock market fluctuates based on supply and demand.", "category": "economics" },
    { "id": "rec30", "chunk_text": "A compass needle points toward the magnetic north pole.", "category": "navigation" },
    { "id": "rec31", "chunk_text": "The universe is expanding, according to the Big Bang theory.", "category": "astronomy" },
    { "id": "rec32", "chunk_text": "Elephants have excellent memory and strong social bonds.", "category": "biology" },
    { "id": "rec33", "chunk_text": "The violin is a string instrument commonly used in orchestras.", "category": "music" },
    { "id": "rec34", "chunk_text": "The heart pumps blood throughout the human body.", "category": "biology" },
    { "id": "rec35", "chunk_text": "Ice cream melts when exposed to heat.", "category": "food science" },
    { "id": "rec36", "chunk_text": "Solar panels convert sunlight into electricity.", "category": "technology" },
    { "id": "rec37", "chunk_text": "The French Revolution began in 1789.", "category": "history" },
    { "id": "rec38", "chunk_text": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history" },
    { "id": "rec39", "chunk_text": "Rainbows are caused by light refracting through water droplets.", "category": "physics" },
    { "id": "rec40", "chunk_text": "Mount Everest is the tallest mountain in the world.", "category": "geography" },
    { "id": "rec41", "chunk_text": "Octopuses are highly intelligent marine creatures.", "category": "biology" },
    { "id": "rec42", "chunk_text": "The speed of sound is around 343 meters per second in air.", "category": "physics" },
    { "id": "rec43", "chunk_text": "Gravity keeps planets in orbit around the sun.", "category": "astronomy" },
    { "id": "rec44", "chunk_text": "The Mediterranean diet is considered one of the healthiest in the world.", "category": "nutrition" },
    { "id": "rec45", "chunk_text": "A haiku is a traditional Japanese poem with a 5-7-5 syllable structure.", "category": "literature" },
    { "id": "rec46", "chunk_text": "The human body is made up of about 60% water.", "category": "biology" },
    { "id": "rec47", "chunk_text": "The Industrial Revolution transformed manufacturing and transportation.", "category": "history" },
    { "id": "rec48", "chunk_text": "Vincent van Gogh painted Starry Night.", "category": "art" },
    { "id": "rec49", "chunk_text": "Airplanes fly due to the principles of lift and aerodynamics.", "category": "physics" },
    { "id": "rec50", "chunk_text": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy" }
]

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def pc_db():
    index_name = "learning"

    # Check if the index already exists, otherwise create it
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
            
        )

    # Connect to the index
    index = pc.Index(index_name)

    return index




index = pc_db()

index.upsert_records("learning-namespace", data)

time.sleep(10)

print("Data has upserted")


# View stats for the index
print(index.describe_index_stats())


