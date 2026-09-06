# Running INCEpTION on Minerva

How the self-hosted INCEpTION annotation server was installed, started, and used
on the Minerva HPC cluster. INCEpTION runs on a **compute node** — never the
login node — inside an interactive LSF job, and is reached from a laptop browser
through an SSH tunnel.

---

## Installation

Done once, on the **login node**. Compute nodes have no outbound internet, but
login nodes do, and all filesystems are mounted everywhere — so the JAR is
fetched on the login node and launched later from a compute node.

```bash
mkdir -p ~/inception-local && cd ~/inception-local
wget https://github.com/inception-project/inception/releases/download/inception-40.3/inception-app-webapp-40.3-standalone.jar
ls -lh
```

The file is a few hundred megabytes; if `ls -lh` shows a few kilobytes the
download broke. GitHub returned a 502 on the first attempt, so a retry may be
needed. If the release-tag URL changes, take the executable-JAR link from
<https://inception-project.github.io/downloads/> instead.

Check the home quota first — INCEpTION's embedded database grows over time:

```bash
quota -s 2>/dev/null || df -h ~
```

| | |
|---|---|
| JAR | `~/inception-local/inception-app-webapp-40.3-standalone.jar` |
| Data directory | `~/.inception/` — the database is in `~/.inception/db` and persists between sessions. It is **not** inside `inception-local/`. |
| Java module | `java/21.0.4`. The default `module load java` gives Java 8, which will not work. |
| Server port | `8123`, set via `SERVER_PORT` |
| LSF account | `acc_lothar_lab` |

---

## Starting a session

### 1. SSH into Minerva (Mount Sinai VPN required)

```bash
ssh rademt02@minerva.hpc.mssm.edu
```

### 2. Start tmux, so a dropped connection does not kill the server

```bash
tmux new -s inception
```

Reattach after a drop with `tmux attach -t inception`; detach deliberately with
Ctrl-B then D.

### 3. Request an interactive compute node

```bash
bsub -P acc_lothar_lab -q interactive -n 2 -W 6:00 -R rusage[mem=8000] -Is /bin/bash
```

Wait for `<<Starting on NODE>>` and **write down the node name** — it is needed
for the tunnel, and it differs every session.

### 4. Launch the server, on the compute node

```bash
module load java/21.0.4
cd ~/inception-local
export SERVER_PORT=8123
java -jar inception-app-webapp-40.3-standalone.jar
```

Ready after roughly 25 seconds, when it prints `Started INCEpTION in N seconds`.
Leave this terminal alone; it is the running server.

### 5. Open the SSH tunnel, from a new terminal on the laptop

Replace `NODE` with the compute node from step 3:

```bash
ssh -L 8200:NODE:8123 rademt02@minerva.hpc.mssm.edu
```

Leave it open; it carries the connection. If the local port is in use, pick
another (`8201:NODE:8123`) and browse to that one instead.

**The tunnel must originate on the laptop.** Running the same command from a
Minerva login node opens the port there, where the browser cannot reach it.

### 6. Open in the browser

`http://localhost:8200`, logging in as `admin`.

---

## Shutting down

1. Ctrl-C in the server terminal, and wait for the shutdown messages.
2. `exit` to end the interactive job and release the compute node.
3. Close the tunnel terminal on the laptop.

Nothing is lost: annotations live in `~/.inception/db` on disk. Before a long
break, export the project to `.zip` from Settings as a second copy.

---

## Project configuration

**Span layer.** The built-in *Named entity* layer, internal name
`de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity`, which matches what a
WebAnno TSV3 export writes. Its `value` feature is left as free-text String with
no tagset, so `DISEASE`, `MEDICATION` and `PROCEDURE` import without complaint.

**Granularity: `Token-level`.** This permits multi-token spans. `Single tokens
only` is the restrictive setting and is not what you want.

**Assertion layer.** The gold assertion pass uses a custom layer,
`webanno.custom.Assertion`, with five features — polarity, certainty,
temporality, experiencer, allergy. `rewrite_tsv_layer.py` re-types exported TSVs
onto it with the defaults baked in.

**Recommenders.** INCEpTION enables a string-matching recommender by default,
which suggests spans as you annotate. This is material for a reference standard:
disable it, or record which documents were annotated with it active.

---

## Loading documents

Two paths were used, for different purposes.

**Plain text, for the gold set.** Documents were imported as plain text so that
no model output was ever presented for correction. This is what makes the gold
set an independent reference standard rather than a measure of
agreement-with-the-model.

**WebAnno TSV3, for machine-assisted review.** `import_preannotations.py` POSTs
each TSV to `/api/aero/v1/projects/{id}/documents` with `format=ctsv3`. This
path was built and proven, then deliberately abandoned: correcting 20,937 notes
by hand is not feasible, so the teacher output became training data directly.
The importer is retained because it is what a corrected-silver workflow would
need.

To use it, enable the AERO remote API in `settings.properties` and give the
importing user **ROLE_REMOTE** under Admin → Users. Without the role the API
returns 403. Then find the numeric project id — the URL slug is not the API id:

```bash
curl -s -u "admin:$PW" http://localhost:8123/api/aero/v1/projects | python -m json.tool
```

```bash
read -rs PW    # keeps the password out of shell history
python import_preannotations.py \
  --base-url http://localhost:8123 --user admin --password "$PW" \
  --project-id 3 \
  --tsv-dir annotations/<batch>/inception_webanno_tsv3
```

The import must run on the **same node as the server**, since `localhost:8123`
resolves only there. `bjobs -w` shows which node the job holds.

---

## Gotchas

**Use an ASCII-only admin password.** `curl -u` and Python `requests` encode
non-ASCII characters differently, so a password containing, say, `§` will
authenticate under curl and 401 under `requests`. Setting an ASCII password
through the browser also re-hashes it as BCrypt and quiets the
`LegacyEmptyPasswordChecker` warning.

---

## Limitation

The server is tied to an interactive job, so it stops when the walltime expires.
That is workable for solo annotation. A persistent instance — which two
annotators computing agreement would need — would require something more durable.
