"""The HTTP surface of the install-record domain: one anonymous endpoint that answers every
report the same way."""

from fastapi import APIRouter, Request, Response

from curvestudio_api.deps import Ctx, Db, client_ip
from curvestudio_api.modules.installs import service
from curvestudio_api.modules.installs.schemas import InstallRecordIn

router = APIRouter(prefix="/installs", tags=["installs"])


@router.post("", status_code=204)
def report_install(body: InstallRecordIn, request: Request, ctx: Ctx, db: Db) -> Response:
    """Store one report from a desktop installation.

    Anonymous: no account is looked up and no audit row is written, because the report names
    nobody and changes no account. Twenty an hour per address. The reply is always the same
    empty 204, so nothing tells whether the installation was already known.
    """
    ctx.install_limiter.check(client_ip(request))
    service.record_install(db, body)
    db.commit()
    return Response(status_code=204)
