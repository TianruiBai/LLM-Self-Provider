# D4 — Convenience targets. All targets assume `docker compose` v2.

.PHONY: up down logs restart pull migrate backup rotate-keys ps health smoke build-llama

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f gateway

restart:
	docker compose up -d --force-recreate

pull:
	docker compose pull

ps:
	docker compose ps

health:
	@curl -fsS http://127.0.0.1:8088/health || echo "gateway is not responding"

build-llama:
	docker compose --profile build-llama build llama-server

migrate:
	docker compose exec gateway python -m provider.scripts.migrate_mongo_to_lance \
		--mongo-uri $${MONGO_URI:-mongodb://host.docker.internal:27017} \
		--database $${MONGO_DB:-provider_rag} \
		--collection $${MONGO_COLL:-documents} \
		--lance-dir /app/data/lance \
		--embedding-dim $${EMBED_DIM:-4096}

backup:
	bash scripts/backup.sh

rotate-keys:
	@echo "Rotate PROVIDER_AUTH_PEPPER and PROVIDER_MASTER_KEY in .env, then run:"
	@echo "  docker compose up -d --force-recreate"
	@echo "Existing TOTP secrets are encrypted with the master key and must be re-enrolled after rotation."

# smoke tests import `provider.*`, so they must run from the parent of provider/.
smoke:
	cd .. && python -m provider.tests.smoke_phase_a8
	cd .. && python -m provider.tests.smoke_phase_b1
	cd .. && python -m provider.tests.smoke_phase_b6
	cd .. && python -m provider.tests.smoke_phase_c
