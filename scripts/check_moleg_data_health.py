"""Read-only effective-config / search / DB checks; never print credential values."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


class DataHealth:
    def __init__(self,root):
        from dotenv import load_dotenv
        load_dotenv(root/'operator.env',override=True)
        required=['OPENSEARCH_URL','OPENSEARCH_PASSWORD','POSTGRES_HOST','POSTGRES_PASSWORD']
        if any(not os.environ.get(k) for k in required):
            raise RuntimeError('effective_search_or_database_configuration_missing')
        sys.path.insert(0,str(root/'pod_source/src'))
        from config import settings
        self.settings=settings

    def database(self):
        import psycopg2
        s=self.settings
        connection=psycopg2.connect(host=s.POSTGRES_HOST,port=s.POSTGRES_PORT,dbname=s.POSTGRES_DB,
            user=s.POSTGRES_USER,password=s.POSTGRES_PASSWORD,connect_timeout=5)
        try:
            connection.set_session(readonly=True,autocommit=True)
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
                return cursor.fetchone()==(1,)
        finally:connection.close()

    async def check(self,full=False):
        import httpx
        s=self.settings
        row={}
        try:
            async with httpx.AsyncClient(timeout=5,trust_env=False,verify=s.OPENSEARCH_VERIFY_CERTS,
                auth=s.OPENSEARCH_AUTH) as client:
                response=await client.get(s.OPENSEARCH_URL.rstrip('/')+'/_cluster/health')
                response.raise_for_status()
                row['search_cluster_status']=response.json()['status']
                if full:
                    counts=[]
                    for index in [s.INDEX_NAME_ARTICLE,s.INDEX_NAME_WORLD_LAW_ORIGIN,s.TITLE_INDEX_NAME]:
                        response=await client.get(s.OPENSEARCH_URL.rstrip('/')+'/'+index+'/_count')
                        response.raise_for_status();counts.append(response.json()['count'])
                    row['required_index_document_counts']=counts
            row['database_select_one']=await asyncio.to_thread(self.database)
            row['ok']=row['search_cluster_status'] in ['green','yellow'] and row['database_select_one']
            if full:row['ok']=row['ok'] and all(n>0 for n in row['required_index_document_counts'])
        except Exception as exc:
            row.update(ok=False,error_type=type(exc).__name__)
        return row


async def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    row=await DataHealth(args.directory.resolve()).check(full=True)
    with (args.directory/'data-health-preflight.json').open('x') as sink:
        json.dump(row,sink,indent=2);sink.write('\n')
    print(json.dumps(row))
    if not row['ok']:raise SystemExit(1)


if __name__=='__main__':asyncio.run(main())
