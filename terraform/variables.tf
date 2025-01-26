variable "region" {
  description = "The GCP region"
  default     = "asia-south1" # Mumbai region for low latency
}

variable "zone" {
  description = "The GCP zone"
  default     = "asia-south1-a"
}

variable "project_id" {
  description = "The GCP project ID"
}

variable "container_image" {
  description = "The Docker image for the fraud detection app"
}