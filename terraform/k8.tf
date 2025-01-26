resource "kubernetes_deployment" "fraud_detection" {
  metadata {
    name = "fraud-detection-deployment"
    labels = {
      "app" = "fraud-detection"
    }
  }

  spec {
    replicas = 2
    selector {
      match_labels = {
        "app" = "fraud-detection"
      }
    }

    template {
      metadata {
        labels = {
          "app" = "fraud-detection"
        }
      }

      spec {
        container {
          name  = "fraud-detection-container"
          image = var.container_image
          port {
            container_port = 80
          }
        }
      }
    }
  }
}

resource "google_compute_address" "default" {
  name   = "fraud-detection-ip"
  region = var.region
}

resource "kubernetes_service" "fraud_detection_service" {
  metadata {
    name = "fraud-detection-service"
  }

  spec {
    type             = "LoadBalancer"
    load_balancer_ip = google_compute_address.default.address
    port {
      port        = 80
      target_port = 80
    }
    selector = {
      "app" = "fraud-detection"
    }
  }
}