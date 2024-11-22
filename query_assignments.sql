SELECT 
a.key.id as assignment_id,
a.value.title as assignment_title,
a.value.description as assignment_description
from `udp-umich-prod.canvas.assignments` a
where 
a.value.context_id = @course_id
and a.value.workflow_state='published'
order by a.key.id