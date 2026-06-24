from graphiti_core.nodes import EpisodeType
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from dotenv import load_dotenv
from logging import INFO
from pathlib import Path
from graphiti_core.utils.bulk_utils import RawEpisode
import os
import logging
import json
import asyncio


logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).with_name('.env'), override=True)

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
neo4j_database = os.environ.get('NEO4J_DATABASE', neo4j_user)

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Initialize Graphiti with Neo4j connection
    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    graphiti = Graphiti(graph_driver=driver)

    logger.info("Adding episodes as unstructured text")
    await graphiti.add_episode(
        name="tech_innovation_article",
        episode_body=(
            "MIT researchers have unveiled 'ClimateNet', an AI system capable of predicting "
            "climate patterns with unprecedented accuracy. Early tests show it can forecast "
            "major weather events up to three weeks in advance, potentially revolutionizing "
            "disaster preparedness and agricultural planning."
        ),
        source=EpisodeType.text,
        # A description of the source (e.g., "podcast", "news article")
        source_description="Technology magazine article",
        # The timestamp for when this episode occurred or was created
        reference_time=datetime(2023, 11, 15, 9, 30),
    )

    logger.info("Adding episodes as structured JSON data")
    product_data = {
        "id": "PROD001",
        "name": "Men's SuperLight Wool Runners",
        "color": "Dark Grey",
        "sole_color": "Medium Grey",
        "material": "Wool",
        "technology": "SuperLight Foam",
        "price": 125.00,
        "in_stock": True,
        "last_updated": "2024-03-15T10:30:00Z"
    }

    # Add the episode to the graph
    await graphiti.add_episode(
        name="Product Update - PROD001",
        episode_body=json.dumps(product_data),  # episode_body must be a JSON string, not a dict
        source=EpisodeType.json,
        source_description="Allbirds product catalog update",
        reference_time=datetime.now(),
    )

    logger.info("Adding multiple episodes in bulk")
    product_data = [
        {
            "id": "PROD001",
            "name": "Men's SuperLight Wool Runners",
            "color": "Dark Grey",
            "sole_color": "Medium Grey",
            "material": "Wool",
            "technology": "SuperLight Foam",
            "price": 125.00,
            "in_stock": True,
            "last_updated": "2024-03-15T10:30:00Z"
        },
        {
            "id": "PROD0100",
            "name": "Kids Wool Runner-up Mizzles",
            "color": "Natural Grey",
            "sole_color": "Orange",
            "material": "Wool",
            "technology": "Water-repellent",
            "price": 80.00,
            "in_stock": True,
            "last_updated": "2024-03-17T14:45:00Z"
        }
    ]

    # Prepare the episodes for bulk loading

    bulk_episodes = [
        RawEpisode(
            name=f"Product Update - {product['id']}",
            content=json.dumps(product),
            source=EpisodeType.json,
            source_description="Allbirds product catalog update",
            reference_time=datetime.now()
        )
        for product in product_data
    ]

    await graphiti.add_episode_bulk(bulk_episodes)

    logger.info("Finished adding episodes examples")

if __name__ == "__main__":

    asyncio.run(main())