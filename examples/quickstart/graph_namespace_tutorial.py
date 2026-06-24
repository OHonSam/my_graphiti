from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from dotenv import load_dotenv
from logging import INFO
from pathlib import Path
import os
import logging
import asyncio
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.nodes import EpisodeType
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
import uuid
from datetime import datetime
# For more advanced node-specific searches, use the _search method with a recipe
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF

# Custom Entity Types
class Person(BaseModel):
    """A person entity with biographical information."""
    age: Optional[int] = Field(None, description="Age of the person")
    occupation: Optional[str] = Field(None, description="Current occupation")
    location: Optional[str] = Field(None, description="Current location")
    birth_date: Optional[datetime] = Field(None, description="Date of birth")

    @field_validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 150):
            raise ValueError('Age must be between 0 and 150')
        return v

class Company(BaseModel):
    """A business organization."""
    industry: Optional[str] = Field(None, description="Primary industry")
    founded_year: Optional[int] = Field(None, description="Year company was founded")
    headquarters: Optional[str] = Field(None, description="Location of headquarters")
    # employee_count: Optional[int] = Field(None, description="Number of employees")

class Product(BaseModel):
    """A product or service."""
    category: Optional[str] = Field(None, description="Product category")
    price: Optional[float] = Field(None, description="Price in USD")
    release_date: Optional[datetime] = Field(None, description="Product release date")

# Custom Edge Types
class Employment(BaseModel):
    """Employment relationship between a person and company."""
    position: Optional[str] = Field(None, description="Job title or position")
    start_date: Optional[datetime] = Field(None, description="Employment start date")
    end_date: Optional[datetime] = Field(None, description="Employment end date")
    salary: Optional[float] = Field(None, description="Annual salary in USD")
    is_current: Optional[bool] = Field(None, description="Whether employment is current")

class Investment(BaseModel):
    """Investment relationship between entities."""
    amount: Optional[float] = Field(None, description="Investment amount in USD")
    investment_type: Optional[str] = Field(None, description="Type of investment (equity, debt, etc.)")
    stake_percentage: Optional[float] = Field(None, description="Percentage ownership")
    investment_date: Optional[datetime] = Field(None, description="Date of investment")

class Partnership(BaseModel):
    """Partnership relationship between companies."""
    partnership_type: Optional[str] = Field(None, description="Type of partnership")
    duration: Optional[str] = Field(None, description="Expected duration")
    deal_value: Optional[float] = Field(None, description="Financial value of partnership")


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


async def main():
    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
        database=neo4j_database,
    )
    graphiti = Graphiti(graph_driver=driver)

    await graphiti.add_episode(
        name="customer_interaction",
        episode_body="Customer Jane mentioned she loves our new SuperLight Wool Runners in Dark Grey.",
        source=EpisodeType.text,
        source_description="Customer feedback",
        reference_time=datetime.now(),
        group_id="customer_team"  # This namespaces the episode and its entities
    )
    logger.info("Added customer interaction episode with namespace 'customer_team'")

    # Define a namespace for this data
    namespace = "product_catalog"

    # Create source and target nodes with the namespace
    source_node = EntityNode(
        uuid=str(uuid.uuid4()),
        name="SuperLight Wool Runners",
        group_id=namespace  # Apply namespace to source node
    )

    target_node = EntityNode(
        uuid=str(uuid.uuid4()),
        name="Sustainable Footwear",
        group_id=namespace  # Apply namespace to target node
    )

    # Create an edge with the same namespace
    edge = EntityEdge(
        group_id=namespace,  # Apply namespace to edge
        source_node_uuid=source_node.uuid,
        target_node_uuid=target_node.uuid,
        created_at=datetime.now(),
        name="is_category_of",
        fact="SuperLight Wool Runners is a product in the Sustainable Footwear category"
    )

    # Add the triplet to the graph
    await graphiti.add_triplet(source_node, edge, target_node)
    logger.info("Added triplet with namespace 'product_catalog'")

    # Search within a specific namespace
    search_results = await graphiti.search(
        query="Wool Runners",
        group_ids=["product_catalog"]  # Only search within this namespace
    )

    # Create a search config for nodes only
    node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    node_search_config.limit = 5  # Limit to 5 results

    # Execute the node search within a specific namespace
    node_search_results = await graphiti._search(
        query="SuperLight Wool Runners",
        group_ids=["product_catalog"],  # Only search within this namespace
        config=node_search_config
    )

    logger.info("Search results for 'SuperLight Wool Runners' within 'product_catalog' namespace: %s", node_search_results.nodes)

    async def add_customer_data(tenant_id, customer_data):
        """Add customer data to a tenant-specific namespace"""
        
        # Use the tenant_id as the namespace
        namespace = f"tenant_{tenant_id}"
        
        # Create an episode for this customer data
        await graphiti.add_episode(
            name=f"customer_data_{customer_data['id']}",
            episode_body=customer_data,
            source=EpisodeType.json,
            source_description="Customer profile update",
            reference_time=datetime.now(),
            group_id=namespace  # Namespace by tenant
        )

    async def search_tenant_data(tenant_id, query):
        """Search within a tenant's namespace"""
        
        namespace = f"tenant_{tenant_id}"
        
        # Only search within this tenant's namespace
        return await graphiti.search(
            query=query,
            group_ids=[namespace]
        )


if __name__ == "__main__":
    asyncio.run(main())