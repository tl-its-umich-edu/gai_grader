SELECT 
a.key.id as assignment_id,
a.value.title as assignment_title,
a.value.description as assignment_description
from `udp-umich-prod.canvas.assignments` a
where 
a.value.context_id = 656488
and a.value.workflow_state='published'
order by a.key.id