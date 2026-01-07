from django.urls import path
from . import views
from .views import export_actors_jsonl
from .views import qdrant_admin


urlpatterns = [
    path("", views.index, name="index"),
    path("upload", views.upload_page, name="upload_page"),
    path("api/chat", views.api_chat, name="api_chat"),
    path("api/ask", views.api_ask, name="api_ask"),
    path("api/math", views.api_math, name="api_math"),

    path("api/upload-jsonl", views.api_upload_jsonl, name="api_upload_jsonl"),
    path("export/actors.jsonl", export_actors_jsonl),

    path("admin-tools/qdrant/", qdrant_admin, name="qdrant_admin"),



]
