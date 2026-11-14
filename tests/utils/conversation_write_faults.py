"""Native uniqueness failure after the real first conversation INSERT."""

from dataclasses import dataclass, field

from kestrel_sovereign.conversation_speakers import EXTERNAL_ASSISTANT_ROLE
from kestrel_sovereign.storage.db.interface import QueryError


@dataclass
class ConversationInsertFault:
    user_row_ids: list[int] = field(default_factory=list)
    native_errors: list[QueryError] = field(default_factory=list)


def fail_external_conversation_insert(store, monkeypatch):
    """Keep the canonical writer and force an actual duplicate primary key."""
    original = store._insert_message
    fault = ConversationInsertFault()

    async def reject_second(**arguments):
        if arguments["role"] == EXTERNAL_ASSISTANT_ROLE:
            assert len(fault.user_row_ids) == 1
            try:
                await store.db.execute(
                    "INSERT INTO conversation_history (id,agent_id,role,content) "
                    "SELECT id,agent_id,role,content FROM conversation_history WHERE id=?",
                    (fault.user_row_ids[0],))
            except QueryError as error:
                fault.native_errors.append(error)
                raise
            raise AssertionError("Owned native duplicate row unexpectedly succeeded")
        receipt = await original(**arguments)
        if arguments["role"] == "user":
            fault.user_row_ids.append(receipt.row_id)
        return receipt

    monkeypatch.setattr(store, "_insert_message", reject_second)
    return fault
