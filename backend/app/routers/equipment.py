import os
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from loguru import logger
from bson import ObjectId

from app.database import get_database
from app.models.equipment import Equipment
from app.models.document import Document
from app.config import settings



router = APIRouter()


@router.post("/", response_model=Equipment, status_code=status.HTTP_201_CREATED)
async def create_equipment(equipment: Equipment):
    """Create a new equipment"""
    db = get_database()
    
    # Check if equipment name already exists
    existing = await db.equipment.find_one({"name": equipment.name, "tenant_id": equipment.tenant_id})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Equipment with this name already exists"
        )
    
    # Add timestamps
    now = datetime.utcnow()
    equipment_dict = equipment.model_dump(exclude={"id"}, exclude_none=True)
    equipment_dict["created_at"] = now
    equipment_dict["updated_at"] = now
    
    # Insert into database
    result = await db.equipment.insert_one(equipment_dict)
    # Create response with _id as string
    response_dict = equipment.model_dump(exclude={"id"}, exclude_none=True)
    response_dict["_id"] = str(result.inserted_id)
    return Equipment(**response_dict)


@router.get("/", response_model=List[Equipment], status_code=status.HTTP_200_OK)
async def get_equipment():
    """Get all equipment"""
    db = get_database()
    equipment_list = await db.equipment.find({}).to_list(length=None)
    # Convert ObjectId to string for _id field
    result = []
    for item in equipment_list:
        item_dict = dict(item)
        if '_id' in item_dict and isinstance(item_dict['_id'], ObjectId):
            item_dict['_id'] = str(item_dict['_id'])
        result.append(Equipment(**item_dict))
    return result


@router.get("/{equipment_id}", response_model=Equipment, status_code=status.HTTP_200_OK)
async def get_one_equipment(equipment_id: str):
    """Get an equipment by ID"""
    db = get_database()
    equipment = await db.equipment.find_one({"_id": ObjectId(equipment_id)})
    if not equipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Equipment not found"
        )
    equipment_dict = dict(equipment)
    if '_id' in equipment_dict and isinstance(equipment_dict['_id'], ObjectId):
        equipment_dict['_id'] = str(equipment_dict['_id'])
    return Equipment(**equipment_dict)


@router.get("/{equipment_id}/documents", status_code=status.HTTP_200_OK)
async def list_equipment_documents(equipment_id: str):
    """List all documents for an equipment"""
    db = get_database()
    
    # Verify equipment exists
    equipment = await db.equipment.find_one({"_id": ObjectId(equipment_id)})
    if not equipment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Equipment not found"
        )
    
    documents = await db.documents_metadata.find({
        "equipment_id": ObjectId(equipment_id),
        "is_disabled": {"$ne": True}
    }).to_list(length=1000)
    
    # Convert ObjectId and datetime fields to strings for JSON serialization
    serialized_documents = []
    for doc in documents:
        doc_dict = dict(doc)
        if '_id' in doc_dict and isinstance(doc_dict['_id'], ObjectId):
            doc_dict['_id'] = str(doc_dict['_id'])
        if 'equipment_id' in doc_dict and isinstance(doc_dict['equipment_id'], ObjectId):
            doc_dict['equipment_id'] = str(doc_dict['equipment_id'])
        if 'created_at' in doc_dict and isinstance(doc_dict['created_at'], datetime):
            doc_dict['created_at'] = doc_dict['created_at'].isoformat()
        if 'updated_at' in doc_dict and isinstance(doc_dict['updated_at'], datetime):
            doc_dict['updated_at'] = doc_dict['updated_at'].isoformat()
        serialized_documents.append(doc_dict)
    
    return {"documents": serialized_documents, "count": len(serialized_documents)}

