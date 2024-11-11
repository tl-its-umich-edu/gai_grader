    SELECT ra.value.association_type, 
    ra.value.rubric_id, 
    r.value.data, 
    r.value.title, 
    r.value.context_type, 
    r.value.points_possible, 
    r.value.workflow_state
    FROM 
    `udp-umich-prod.canvas.rubric_associations` as ra, 
    `udp-umich-prod.canvas.rubrics` as r
    where ra.value.context_type = 'Course'
    and ra.value.context_id =656488
    and ra.value.workflow_state='active'
    and ra.value.rubric_id = r.key.id
    order by r.value.title