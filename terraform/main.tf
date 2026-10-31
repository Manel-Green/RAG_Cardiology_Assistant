terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = "manel-portfolio-demos"
  region  = "europe-west3"
}

resource "google_storage_bucket" "rag_heart_studies" {
  name     = "manel-portfolio-demos-rag-heart-studies"
  location = "europe-west3"
}

resource "google_document_ai_processor" "rag_heart_studies_ocr" {
  location     = "eu"
  display_name = "rag-heart-studies-ocr"
  type         = "OCR_PROCESSOR"
}

resource "google_vertex_ai_index" "rag_heart_studies" {
  region       = "europe-west3"
  display_name = "rag-heart-studies-index-small"
  description  = "Tree-AH index (small shard) over heart-studies RAG chunk embeddings"

  metadata {
    contents_delta_uri = "gs://manel-portfolio-demos-rag-heart-studies/processed/embeddings/vectors/"

    config {
      dimensions                  = 768
      approximate_neighbors_count = 10
      distance_measure_type       = "COSINE_DISTANCE"
      feature_norm_type           = "NONE"

      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 500
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [
      metadata[0].contents_delta_uri,
      metadata[0].config[0].feature_norm_type,
    ]
  }
}

resource "google_vertex_ai_index_endpoint" "rag_heart_studies" {
  region                  = "europe-west3"
  display_name            = "rag-heart-studies-endpoint"
  public_endpoint_enabled = true
  lifecycle {
    ignore_changes = [public_endpoint_enabled]
  }
}

resource "google_cloud_run_v2_service" "rag_heart_studies_backend" {
  name     = "rag-heart-studies-backend"
  location = "europe-west3"
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "europe-west3-docker.pkg.dev/manel-portfolio-demos/cloud-run-source-deploy/rag-heart-studies-backend:01797dbe462dd4390e2f00d9c666ac1015e9b1bd"

      ports {
        container_port = 8080
      }

       resources {
        limits = {
          cpu    = "1000m"
          memory = "512Mi"
        }
        cpu_idle          = true
        startup_cpu_boost = true
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 20
    }

    max_instance_request_concurrency = 80
    timeout                          = "300s"
  }
}


resource "google_cloudbuild_trigger" "rag_backend_deploy" {
  name     = "rag-backend-deploy"
  location = "europe-west3"
  filename = "cloudbuild.yaml"

  repository_event_config {
    repository = "projects/manel-portfolio-demos/locations/europe-west3/connections/github-manel-green/repositories/Manel-Green-RAG_Cardiology_Assistant"
    push {
      branch = "^main$"
    }
  }

  service_account = "projects/manel-portfolio-demos/serviceAccounts/90354696252-compute@developer.gserviceaccount.com"
}

resource "google_cloud_run_v2_service_iam_member" "rag_heart_studies_public" {
  project  = google_cloud_run_v2_service.rag_heart_studies_backend.project
  location = google_cloud_run_v2_service.rag_heart_studies_backend.location
  name     = google_cloud_run_v2_service.rag_heart_studies_backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}